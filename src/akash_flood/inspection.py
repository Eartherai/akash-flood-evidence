"""Inspection policies with no access to unobserved road states.

The uncertainty unit can be a jointly closed OSM bridge way or a physical
segment. This is a finite graph experiment; it infers no flood or passability.
"""
from collections import deque
import random

import networkx as nx

from .access import _snap
from .access_bounds import _edge_index


class InspectionExperiment:
    def __init__(self, graph, settlements, destinations, groups, max_snap_m=500):
        self.graph = graph
        self.edge_index = _edge_index(graph)
        self.groups = {key: frozenset(value) for key,value in sorted(groups.items())}
        assigned = set()
        for key,edges in self.groups.items():
            if not edges or not edges <= self.edge_index.keys() or edges & assigned:
                raise ValueError('Inspection groups must be nonempty, known and disjoint')
            assigned |= edges
        self.keys = tuple(self.groups)
        self.all_mask = (1 << len(self.keys))-1
        self.edge_masks = {edge:1<<i for i,key in enumerate(self.keys) for edge in self.groups[key]}
        self.predecessors = {node:[] for node in graph}
        self.successors = {node:[] for node in graph}
        for edge,(u,v,key,data) in self.edge_index.items():
            mask = self.edge_masks.get(edge,0)
            self.predecessors[v].append((u,mask,edge))
            self.successors[u].append((v,mask,edge))
        for neighbours in [*self.predecessors.values(),*self.successors.values()]:
            neighbours.sort(key=lambda entry:(str(entry[0]),entry[2]))
        self.targets = frozenset(snap['node_id'] for point in destinations if (snap:=_snap(graph,point,max_snap_m))['mapped'])
        self.settlements = settlements
        self.snaps = [_snap(graph,point,max_snap_m) for point in settlements]
        self.reach_cache = {}
        self.state_cache = {}
        self.centrality_cache = {}
        self.baseline_nodes,self.baseline_routes = self._reach(0)
        self.baseline_bits = sum(1<<i for i,snap in enumerate(self.snaps) if snap['mapped'] and snap['node_id'] in self.baseline_nodes)
        self.initial_ambiguous_bits = self.state(0,0)['ambiguous_bits']

    def _reach(self, removed):
        if removed in self.reach_cache:
            return self.reach_cache[removed]
        seen = set(self.targets)
        queue = deque(sorted(seen,key=str))
        next_edge = {}
        while queue:
            v = queue.popleft()
            for u,mask,edge in self.predecessors[v]:
                if u not in seen and not mask & removed:
                    seen.add(u); queue.append(u); next_edge[u]=(v,edge)
        self.reach_cache[removed]=(frozenset(seen),next_edge)
        return self.reach_cache[removed]

    def _source_bits(self, reachable):
        return sum(1<<i for i,snap in enumerate(self.snaps) if snap['mapped'] and snap['node_id'] in reachable)

    def state(self, closed, opened):
        if closed & opened or (closed|opened) & ~self.all_mask:
            raise ValueError('Known open/closed states must be disjoint known group bits')
        key = (closed,opened)
        if key in self.state_cache:
            return self.state_cache[key]
        unknown = self.all_mask ^ (closed|opened)
        upper = self._source_bits(self._reach(closed)[0])
        lower = self._source_bits(self._reach(closed|unknown)[0])
        if lower & ~upper:
            raise AssertionError('Monotone reachability violated')
        ambiguous = (upper & ~lower) & self.baseline_bits
        certain_connected = lower & self.baseline_bits
        certain_disconnected = self.baseline_bits & ~upper
        value = {'closed_mask':closed,'opened_mask':opened,'unknown_mask':unknown,
                 'ambiguous_bits':ambiguous,'unresolved_count':ambiguous.bit_count(),
                 'connected_bits':certain_connected,'disconnected_bits':certain_disconnected}
        self.state_cache[key]=value
        return value

    def eligible(self, state):
        # Shared relevance filter: an unknown link must lie on an optimistic
        # walk from a currently ambiguous source to a mapped destination.
        closed = state['closed_mask']
        reach_targets = self._reach(closed)[0]
        sources = {snap['node_id'] for i,snap in enumerate(self.snaps) if state['ambiguous_bits'] & (1<<i)}
        forward = set(sources); queue = deque(sources)
        while queue:
            u = queue.popleft()
            for v,mask,edge in self.successors[u]:
                if v not in forward and not mask & closed:
                    forward.add(v); queue.append(v)
        return [i for i,key in enumerate(self.keys) if state['unknown_mask'] & (1<<i)
                and any(self.edge_index[edge][0] in forward and self.edge_index[edge][1] in reach_targets for edge in self.groups[key])]

    def branch_resolutions(self, state, group_index):
        bit = 1<<group_index
        if not state['unknown_mask'] & bit:
            raise ValueError('Cannot inspect a previously revealed group')
        a = self.state(state['closed_mask'],state['opened_mask']|bit)
        b = self.state(state['closed_mask']|bit,state['opened_mask'])
        current = state['ambiguous_bits']
        return ((current & ~a['ambiguous_bits']).bit_count(),(current & ~b['ambiguous_bits']).bit_count())

    def _route_counts(self, state, candidates):
        _,routes = self._reach(state['closed_mask'])
        counts = {i:0 for i in candidates}
        for i,snap in enumerate(self.snaps):
            if not state['ambiguous_bits'] & (1<<i):
                continue
            node = snap['node_id']; used = set()
            while node in routes:
                node,edge = routes[node]
                bit = self.edge_masks.get(edge,0)
                if bit:
                    group = bit.bit_length()-1
                    if group in counts:
                        used.add(group)
            for group in used:
                counts[group]+=1
        return counts

    def _betweenness(self, state, candidates):
        cache_key = (state['closed_mask'],state['ambiguous_bits'])
        if cache_key not in self.centrality_cache:
            graph = self.graph.copy()
            graph.remove_edges_from([(u,v,key) for edge,(u,v,key,data) in self.edge_index.items() if self.edge_masks.get(edge,0)&state['closed_mask']])
            sources = sorted({snap['node_id'] for i,snap in enumerate(self.snaps) if state['ambiguous_bits'] & (1<<i)},key=str)
            centrality = nx.edge_betweenness_centrality_subset(graph,sources,list(self.targets),normalized=False,weight=None)
            self.centrality_cache[cache_key] = {
                i:sum(centrality.get(self.edge_index[edge][:3],0) for edge in self.groups[key])
                for i,key in enumerate(self.keys)}
        return self.centrality_cache[cache_key]

    def choose(self, state, policy, rng=None):
        candidates = self.eligible(state)
        if not candidates:
            return None
        if policy=='maximin':
            values = {i:self.branch_resolutions(state,i) for i in candidates}
            # Primary min(branch) gain; tie by max gain, then source group ID.
            return min(candidates,key=lambda i:(-min(values[i]),-max(values[i]),self.keys[i]))
        if policy=='adaptive_route_count':
            values = self._route_counts(state,candidates)
            return min(candidates,key=lambda i:(-values[i],self.keys[i]))
        if policy=='adaptive_betweenness':
            values = self._betweenness(state,candidates)
            return min(candidates,key=lambda i:(-values[i],self.keys[i]))
        if policy=='random_relevant':
            return (rng or random.Random(0)).choice(candidates)
        raise ValueError('Unknown policy')

    def run(self, hidden_closed_mask, policy, budget=5, random_seed=0):
        if hidden_closed_mask & ~self.all_mask:
            raise ValueError('Hidden state contains unknown group bits')
        closed=opened=0; rng=random.Random(random_seed); trace=[]
        fully_revealed = self.state(hidden_closed_mask,self.all_mask ^ hidden_closed_mask)
        assert fully_revealed['unresolved_count']==0
        for step in range(budget+1):
            state=self.state(closed,opened)
            # Definitive statements must agree with fully revealed graph states.
            assert not state['connected_bits'] & ~fully_revealed['connected_bits']
            assert not state['disconnected_bits'] & ~fully_revealed['disconnected_bits']
            trace.append({'step':step,'unresolved_count':state['unresolved_count'],
                          'resolved_count':self.initial_ambiguous_bits.bit_count()-state['unresolved_count'],
                          'known_closed_mask':closed,'known_open_mask':opened})
            if step==budget or not state['unresolved_count']:
                continue
            action=self.choose(state,policy,rng)
            if action is None:
                continue
            trace[-1]['chosen_group']=self.keys[action]
            trace[-1]['branch_resolutions_open_closed']=self.branch_resolutions(state,action)
            bit=1<<action
            # Hidden state is accessed only after the policy selects its action.
            observed_closed=bool(hidden_closed_mask & bit)
            trace[-1]['observed_closed']=observed_closed
            if observed_closed: closed|=bit
            else: opened|=bit
        return trace
