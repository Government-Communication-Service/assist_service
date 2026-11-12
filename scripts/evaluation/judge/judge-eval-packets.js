export const meta = {
  name: 'judge-eval-packets',
  description: 'Judge anonymised evaluation packets in parallel batches and return validated verdicts',
  whenToUse: 'Run in a judging workspace created by judge.py export; pass the contents of batches.json as args',
  phases: [{ title: 'Judge', detail: 'one agent per batch of packets' }],
}

const VERDICTS_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          a_acceptable: { type: 'boolean' },
          b_acceptable: { type: 'boolean' },
          winner: { enum: ['A', 'B', 'tie'] },
          reason: { type: 'string' },
          judge: { type: 'string' },
        },
        required: ['id', 'a_acceptable', 'b_acceptable', 'winner', 'reason', 'judge'],
      },
    },
  },
  required: ['verdicts'],
}

// args: {batches: [{ids: ['001', '002', ...]}, ...]}, as written to batches.json by judge.py export.
phase('Judge')
const results = await parallel(
  args.batches.map(batch => () =>
    agent(
      `You are judging evaluation packets. First read .claude/skills/eval-judge/SKILL.md and follow its rubric. ` +
        `Judge exactly these packets: ${batch.ids.join(', ')}. For each id, read packets/<id>.md and nothing else: ` +
        `do not read any other file or search the workspace. Judge each packet on its own. ` +
        `Do not write any files: return one verdict per id (the id as given, e.g. "007") in your structured output, ` +
        `and put your own model id in the "judge" field.`,
      { label: `judge ${batch.ids[0]}-${batch.ids.at(-1)}`, phase: 'Judge', schema: VERDICTS_SCHEMA },
    ),
  ),
)

const expected = args.batches.flatMap(batch => batch.ids)
const verdicts = results.filter(Boolean).flatMap(result => result.verdicts)
const judged = new Set(verdicts.map(verdict => verdict.id))
const missing = expected.filter(id => !judged.has(id))
log(`${verdicts.length}/${expected.length} verdicts returned; ${missing.length} missing`)
return { verdicts, missing }
