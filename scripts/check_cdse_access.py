"""Authenticate locally and read account quota metadata; execute no jobs.

Run: uv run --extra satellite python scripts/check_cdse_access.py
The user completes the printed device-login URL in their browser. Tokens are
process-local and are neither printed nor saved. No account identity is saved.
"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path

import openeo

from akash_flood.radar import BACKEND


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll-seconds',type=int,default=300)
    args=parser.parse_args()
    if not 30 <= args.poll_seconds <= 1800:raise SystemExit('Poll window must be between 30 and 1800 seconds')
    connection = openeo.connect(BACKEND)
    connection.authenticate_oidc_device(store_refresh_token=False, max_poll_time=args.poll_seconds)
    account = connection.describe_account()
    # No user_id, name, email or private account profile enters the evidence.
    quota = {key: account[key] for key in ['budget','currency','free_credits','credits','credits_remaining','quota','quotas'] if key in account}
    evidence = {
        'checked_at':datetime.now(timezone.utc).isoformat(),
        'backend':BACKEND,'authenticated':True,'account_endpoint_accessible':True,
        'quota_metadata':quota,'processing_started':False,'tokens_persisted':False,
        'interpretation':'Authentication/read-only access only. Neither image availability nor processing cost has been established.',
    }
    target = Path(__file__).resolve().parents[1]/'data/rasuwa-pilot/s1-jobs/account-access.json'
    target.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))
    print('No processing job was created. Review the account quota and prepared radar recipe before execution.')


if __name__=='__main__':
    main()
