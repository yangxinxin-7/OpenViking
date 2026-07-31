---
name: experience_loader
description: Load relevant OpenViking experience memories via case-linked experience candidates before solving a task.
---

# experience_loader

Use this skill before taking task actions when reusable execution experience may help.

## Required workflow

1. Before taking task actions, call `search_experience` with a natural-language query that describes the current task.
2. Build the query from the current domain, user intent, target object, requested operation, policy keywords, and likely tool/action family. Avoid vague queries such as "help user".
3. Review the returned candidates. Each candidate is a matched case plus linked experience entries; each experience entry includes its `name`, `uri`, and a short `situation` snippet describing its applicability and exclusions.
4. **Gate before reading.** For each linked experience, read its `situation` snippet and check whether the current task matches the experience's applicability AND does NOT match any of its exclusions / "不适用于" / "does not apply to" items. Skip experiences whose situation explicitly excludes your case (e.g. wrong cabin class, flights already flown, different action family, or different change type). Only call `read_experience` on experiences that plausibly apply after this check. If no experience passes the gate, continue without experience guidance.
5. You may call `search_experience` multiple times with refined keywords. But be selective about reading: read at most the ONE best-matching experience per user intent. If the best match is excluded by its own applicability/exclusion conditions, proceed WITHOUT experience guidance for that intent — do NOT fall through to a weaker, lower-ranked match. A partially-matching experience is more dangerous than no experience: your own policy reasoning on current facts beats guidance written for a different situation.
6. Treat loaded experiences as reusable guidance, not as current-task truth. Current policy, current tool results, and current user facts override prior experience.
7. **Re-verify after reading.** Even after `read_experience`, before acting on the experience, check its full `## Situation` against current facts you have obtained from tools (cabin class, reservation status, flight dates, segment state, etc.). If any "不适用于" / exclusion condition matches the current task now that you have concrete facts, DISCARD the experience and proceed from policy and tool results instead — do NOT apply its Approach or Reflect.
8. Multi-intent tasks (e.g. "cancel, then book", "upgrade then change flight", "refuse a modification then offer a fallback") may legitimately require more than one experience; gate and apply each segment's experience independently. Do not end the task (`done` / `transfer_to_human_agents`) just because one segment's experience reaches a local return marker — check whether the user has a remaining intent.
9. If no linked experience is plausibly relevant after gating, continue without experience guidance.

## Hard rules when applying experiences

**Rule 1 — Experiences provide procedure, never data.** A loaded experience tells you *how* to do something (which steps, which checks, in what order). It never tells you *what* the answer is for the current task. Never copy concrete values out of an experience: flight numbers/IDs, origin/destination airports, dates, cabin classes, prices, baggage counts, payment methods, and passenger details must always come from the current conversation and current tool results. If an experience covers only one segment of a multi-intent task, you must still perform the full search / lookup / verification steps for every other segment exactly as thoroughly as you would without any experience — having an experience for one segment is never a license to shortcut another segment. **The same applies to actions, in both directions:** never perform a state-changing action the current user did not ask for merely because the experience's flow lists it (an unrequested write corrupts the environment and fails the task), and never skip something the current user explicitly asked for merely because the experience's flow does not mention it — the experience defines how to perform the operation; the user defines what must be delivered, including every piece of information they asked to be told.

**Rule 5 — Blocked is a conversation point, not an exit.** When policy forbids the user's requested operation, your next step is ALWAYS to tell the user what is not possible and why, and ask how they would like to proceed — never to escalate or transfer unilaterally. Users frequently have an acceptable fallback (a different operation that policy does allow) which they will only reveal after hearing the denial. Escalate to a human only when the user explicitly asks for it or the policy explicitly mandates human handling for that exact case. An experience's claim that a situation "requires transfer / human handling" is NOT policy — before offering or initiating any transfer suggested by an experience, verify that the actual policy text mandates human handling for the current case; if it does not, or if a human could not change the outcome either (the blocking fact is immutable), simply explain the situation to the user and do not volunteer a transfer they never asked about.

**Rule 4 — Search results are an index, not guidance.** The output of `search_experience` (candidate names, signatures, situation snippets) exists ONLY to decide which single experience to read. It carries no information about what the current task's outcome should be. If you did not `read_experience` an entry, act exactly as if you had never seen it — a list of candidates about refusals/escalations is not evidence that the current task ends that way.

**Rule 3 — Never refuse, deny, or escalate by analogy.** An experience ending in refusal, denial, or escalation/transfer describes what was correct under THAT task's verified facts — it is never evidence that the current task should end the same way. Before taking any refusal/denial/escalation suggested by an experience, independently verify the disqualifying condition against the current task's own tool results (e.g. actually check whether an exemption or entitlement applies, whether the object's state really forbids the operation). If the current facts allow a policy-compliant action — especially an alternative the user has explicitly said they would accept — take that action instead of refusing or escalating.

**Rule 2 — Mandatory intent checklist before ending.** Before calling `done` or `transfer_to_human_agents`, re-read the user's requests across the whole conversation and enumerate every distinct intent (e.g. change flight, cancel reservation, add baggage, refund, compensation, book new trip). For each intent, its status must be one of: completed (required tool writes done and confirmed to the user), blocked (ineligible under policy and this was communicated to the user), or explicitly dropped by the user. If any intent has none of these statuses, do not end — continue working on that intent. A local `RETURN_COMPLETED` from one experience never satisfies this checklist for the other intents. **Scope check within each intent:** if an intent applies to multiple objects ("all my reservations", "every business flight"), re-derive the full list of matching objects from tool results and verify the operation was performed on EVERY one of them — completing a subset is not completed. An experience written for a single object never shrinks the user's stated scope.

## Local return markers in loaded experiences

Experience return markers are **local to the covered intent/subtask**. They are not whole-task success/failure labels and are not automatic permission to call `done`.

- `RETURN_COMPLETED`: the specific intent/subtask covered by this experience has been completed, usually after the required business read/write tool calls and required customer communication. If the user has another independent intent, continue with that next intent instead of ending the conversation.
- `RETURN_BLOCKED(reason="...")`: the covered intent/subtask cannot proceed under the current facts, policy, missing input, refusal boundary, or escalation boundary. Perform any required communication/escalation from the experience, then continue other remaining user intents if they are still actionable.
- `RETURN_NOT_APPLICABLE`: the experience does not match the current facts; discard it and use another applicable experience or current policy/tool facts.

Refusal, no-option, policy-ineligible, missing-input, and `transfer_to_human_agents` branches should be interpreted as `RETURN_BLOCKED(...)` for that local intent, not as whole-task completion. Before ending globally, verify that every user intent is completed, blocked, not applicable, or explicitly transferred/stopped by the user/environment.

## Tools

- `search_experience(query, limit=10)`: searches OpenViking `memories/cases` under the current user, reads each matched case's `## Linked Experiences` section, and returns JSON candidates with case score, case URI, task signature, input summary, and linked experience entries (each with `name`, `uri`, and a `situation` snippet from the experience's `## Situation` section).
- `read_experience(experience_uri)`: reads one OpenViking experience memory by full URI and returns Markdown.
