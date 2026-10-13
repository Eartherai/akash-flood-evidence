(() => {
  'use strict';
  const form = document.getElementById('fresh-ask-form');
  const question = document.getElementById('fresh-question');
  const language = document.getElementById('fresh-language');
  const submit = document.getElementById('fresh-ask-button');
  const status = document.getElementById('fresh-model-status');
  const result = document.getElementById('fresh-answer');
  const examples = [...document.querySelectorAll('[data-fresh-question]')];
  let savedAnswer = null;
  for (const button of document.querySelectorAll('[data-fresh-question]')) {
    button.addEventListener('click', () => {
      question.value = button.dataset.freshQuestion;
      language.value = button.dataset.freshLanguage;
      question.focus();
    });
  }
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (submit.disabled || !question.value.trim()) return;
    submit.disabled = true;
    question.disabled = true;
    language.disabled = true;
    for (const button of examples) button.disabled = true;
    result.hidden = true;
    savedAnswer = null;
    status.textContent = 'Checking saved sources and asking the local CPU model…';
    try {
      const response = await fetch('/api/ask', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: question.value, language: language.value}),
      });
      const answer = await response.json();
      if (!response.ok) throw new Error(answer.error || 'The local assistant is unavailable.');
      savedAnswer = answer;
      document.getElementById('fresh-answered-question').textContent = 'Asked: ' + answer.question;
      document.getElementById('fresh-answer-status').textContent = answer.status === 'decline' ? 'This claim is not established' : 'Answer from saved evidence';
      document.getElementById('fresh-answer-text').textContent = answer.answer;
      const sources = document.getElementById('fresh-source-list');
      sources.replaceChildren();
      for (const claim of answer.claims) {
        const references = claim.source_references && claim.source_references.length ? claim.source_references : [claim];
        for (const reference of references) {
          const row = document.createElement('li');
          row.textContent = [claim.fact_id, reference.source_path || '', reference.source_pointer, 'SHA256 ' + reference.source_sha256].filter(Boolean).join(' | ');
          sources.append(row);
        }
      }
      if (!answer.claims.length) {
        const row = document.createElement('li');
        row.textContent = 'No measurement is asserted in this refusal. Case source bindings are included in the saved answer.';
        sources.append(row);
      }
      const corrected = answer.policy_guard && answer.policy_guard.corrected_model_plan;
      document.getElementById('fresh-answer-audit').textContent = 'Source clauses verified. ' + (corrected ? 'The explicit claim policy corrected the raw model selection. ' : '') + 'Nepali has not received human language review.';
      result.hidden = false;
      status.textContent = 'Completed locally on CPU in ' + answer.request_wall_seconds.toFixed(2) + ' seconds, including source checks.';
    } catch (error) {
      savedAnswer = null;
      result.hidden = true;
      status.textContent = error.message + ' No generated answer is displayed.';
    } finally {
      submit.disabled = false;
      question.disabled = false;
      language.disabled = false;
      for (const button of examples) button.disabled = false;
    }
  });
  document.getElementById('fresh-download-answer').addEventListener('click', () => {
    if (!savedAnswer) return;
    const file = new Blob([JSON.stringify(savedAnswer, null, 2) + '\n'], {type: 'application/json;charset=utf-8'});
    const url = URL.createObjectURL(file);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'akash-case-answer-' + savedAnswer.language + '.json';
    link.click();
    URL.revokeObjectURL(url);
  });
})();
