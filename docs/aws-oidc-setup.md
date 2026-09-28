# AWS setup for the eval gate (OIDC, least privilege, budget alarm)

The `Eval gate` workflow calls Claude on Amazon Bedrock. It authenticates with
**OIDC**: on each run GitHub issues a short-lived signed token, and AWS swaps it
for temporary credentials of one IAM role. No AWS access keys are ever stored
in GitHub, so there is nothing long-lived to leak or rotate.

Replace these placeholders everywhere below:

| Placeholder | Example | Where to find it |
|---|---|---|
| `<ACCOUNT_ID>` | `123456789012` | AWS console, top-right account menu |
| `<REGION>` | `us-east-1` | the region you use for Bedrock |
| `<MODEL_ID>` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock console, see step 0 |

---

## Step 0: Enable the model and confirm its ID

1. AWS console → **Amazon Bedrock** → **Model access** (or **Model catalog**).
   Enable **Claude Haiku 4.5** for your account. Anthropic models may ask for a
   one-time use-case form; an admin does this once, not the CI role.
2. Bedrock → **Cross-region inference** (inference profiles). Find Claude Haiku 4.5
   and copy its **inference profile ID** (e.g. `us.anthropic.claude-haiku-4-5-...`).
   Put it in `judge_model_id` in your suite YAML. The `us.` prefix means the
   request may be served from several US regions.
3. Test locally with your own credentials first:
   `judgekit run suites/example.yaml`

## Step 1: Create the GitHub OIDC identity provider (once per AWS account)

IAM → **Identity providers** → **Add provider**:

- Provider type: **OpenID Connect**
- Provider URL: `https://token.actions.githubusercontent.com`
- Audience: `sts.amazonaws.com`

CLI equivalent:

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com
```

(AWS validates GitHub's certificate itself; older guides that ask for a
thumbprint are out of date.)

## Step 2: Create the role and its trust policy

The trust policy decides **who can assume the role**: only GitHub Actions runs
from *your* repo, and only from pull requests or the `main` branch.

Save as `trust-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": [
            "repo:VigneshReddy23@189483487/judgekit@1393612415:pull_request",
            "repo:VigneshReddy23@189483487/judgekit@1393612415:ref:refs/heads/main"
          ]
        }
      }
    }
  ]
}
```

- `aud` makes sure the token was minted for AWS.
- `sub` names the exact repo and trigger. New GitHub repos use an **immutable
  subject**: `repo:<owner>@<owner-id>/<repo>@<repo-id>:...`. The numeric IDs mean a
  deleted-and-recreated repo with the same name can't reuse your role. Find your
  prefix with `gh api repos/<owner>/<repo>/actions/oidc/customization/sub`
  (the `sub_claim_prefix` field); older repos use `repo:<owner>/<repo>:...`.
- Without the `sub` check, **any** GitHub repo could
  assume your role. This is the most common OIDC misconfiguration.
- `pull_request` covers the PR trigger; `ref:refs/heads/main` covers manual
  runs started from `main`.

```bash
aws iam create-role \
  --role-name judgekit-eval-gate \
  --assume-role-policy-document file://trust-policy.json \
  --max-session-duration 3600
```

## Step 3: Attach a least-privilege permission policy

The permission policy decides **what the role can do**: only `bedrock:InvokeModel`
(which the Converse API uses), only for Claude Haiku 4.5, and only through your
inference profile.

A cross-region inference profile needs two grants: one on the profile itself,
and one on the underlying model in each region the profile may route to. The
condition on the second statement allows the model **only when called through
the profile**.

Save as `bedrock-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeViaInferenceProfile",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:<REGION>:<ACCOUNT_ID>:inference-profile/<MODEL_ID>"
    },
    {
      "Sid": "InvokeUnderlyingModelOnlyThroughProfile",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
      "Condition": {
        "StringEquals": {
          "bedrock:InferenceProfileArn": "arn:aws:bedrock:<REGION>:<ACCOUNT_ID>:inference-profile/<MODEL_ID>"
        }
      }
    }
  ]
}
```

Check the foundation-model ID on the inference profile's detail page, since it
must match what the profile routes to.

```bash
aws iam put-role-policy \
  --role-name judgekit-eval-gate \
  --policy-name bedrock-invoke-haiku \
  --policy-document file://bedrock-policy.json
```

Copy the role ARN (`arn:aws:iam::<ACCOUNT_ID>:role/judgekit-eval-gate`):

```bash
aws iam get-role --role-name judgekit-eval-gate --query Role.Arn --output text
```

## Step 4: Tell GitHub about the role

Repo → **Settings** → **Secrets and variables** → **Actions**:

- **Secrets** tab → New repository secret: `AWS_ROLE_ARN` = the role ARN.
- **Variables** tab (optional) → `AWS_REGION` = `<REGION>` (defaults to `us-east-1`).

(A role ARN is not a credential on its own, but keeping it a secret keeps your
account ID out of logs.)

## Step 5: Set a budget alarm (do this before the first run)

A misconfigured loop or a large suite can spend money quietly. AWS Budgets
emails you before that becomes a surprise.

Billing and Cost Management → **Budgets** → **Create budget**:

1. **Customize** → **Cost budget**, period **Monthly**.
2. Amount: a small number you are comfortable with for a side project.
3. Alerts: **Actual** cost at 50%, 80% and 100%, plus **Forecasted** at 100%,
   sent to your email.

Budgets **alert**; they do not stop spending. Also consider:

- Keep `max_workers` modest in suites, so throttling rather than cost is the limit.
- Check the Bedrock pricing page and add `pricing:` to your suite, so every
  report shows an estimated cost.
- Look at **Cost Explorer**, filtered to service = Bedrock, after your first runs.

## Step 6: Run it

GitHub → **Actions** → **Eval gate** → **Run workflow**. When it finishes,
download **eval-report** from the run's **Artifacts** section and open
`report.html`.

`suites/example.yaml` is **designed to fail** (two demo cases are deliberately
bad), so a red run proves the gate works. Point the workflow at your real
suite for day-to-day use.

## Troubleshooting

| Error | Likely cause |
|---|---|
| `Not authorized to perform sts:AssumeRoleWithWebIdentity` | The `sub` in the trust policy doesn't match: check the immutable-subject prefix (see Step 2), the owner/repo name, or the run came from a branch other than `main` |
| `Could not load credentials from any providers` | Missing `permissions: id-token: write`, or the PR came from a fork |
| `AccessDeniedException ... bedrock:InvokeModel` | The model or profile ARN in the policy doesn't match the `judge_model_id` in the suite |
| `ValidationException ... model identifier is invalid` | Wrong model ID, or model access not enabled in this region |
| `ThrottlingException` | Lower `max_workers`; boto3 already retries with backoff |
