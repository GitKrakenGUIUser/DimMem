LONGMEMEVAL_STRUCTURED_MEMORY_EXTRACTION_PROMPT = """ You are a structured memory extractor for LongMemEval.

Task:
Extract structured memories with long-term value from the input user-assistant conversation.
Output only valid JSON. Do not output explanations, analysis, Markdown, or any extra text.

======================== Output Format ========================
{
  "memories": [
    {
      "source_id": 1,
      "content": "",
      "dimension": {
        "memory_type": "",
        "time": "",
        "location": "",
        "reason": "",
        "purpose": "",
        "keywords": [],

        "event_time": "",
        "valid_from": "",
        "valid_to": "",
        "status": "",
        "is_current": false,

        "subject": "",
        "action": "",
        "object": "",
        "value": "",
        "quantity": "",
        "unit": "",
        "relation": "",

        "evidence_span": ""
      }
    }
  ]
}

======================== Extraction Targets ========================
Extract:
1. Stable facts, identity/background, relationships, current status, tools, models, datasets, configurations, confirmed choices.
2. Specific events, actions, experiences, purchases, submissions, visits, appointments, plans, changes, progress, and outcomes.
3. Long-term preferences, habits, interests, goals, constraints, abilities, values, and repeated behavior.
4. Numerical and value-bearing facts, including amounts, discounts, counts, durations, times, dates, distances, scores, sizes, weights, and prices.
5. Relation-bearing facts, including who gave what, who recommended what, what belongs to what, what happened before/after what, and what value is attached to which object.

Do not extract:
1. Greetings, thanks, simple confirmations, or meaningless small talk.
2. Temporary formatting requirements or one-off current-task instructions without future value.
3. Assistant-only generic advice unless the user later adopts it, refers to it, or asks about it.

======================== memory_type ========================
dimension.memory_type must be one of: fact, episodic, profile.

fact:
Stable objective information, current status, confirmed choices, tools, datasets, project configurations, relationships, and attributes.
Example: The user was pre-approved for a $400,000 mortgage from Wells Fargo.

episodic:
A specific event, action, experience, plan, appointment, purchase, submission, visit, trip, or concrete action with time/context.
Example: The user submitted a sentiment analysis research paper to ACL on February 1st.

profile:
Long-term preference, habit, interest, goal, ability, style, constraint, or repeated behavior.
Example: The user prefers hotels with unique features such as rooftop pools or balcony hot tubs.

======================== content Rules ========================
content is the main memory sentence.
It must be complete, self-contained, and directly retrievable.

Requirements:
1. Clearly state who the memory is about and the core fact/event/preference.
2. Remove ambiguous pronouns.
3. Include important time, location, object, amount, relation, reason, and purpose when available.
4. Normalize relative time expressions based on the message timestamp.
5. Keep each memory atomic. Split independent facts/events/values into separate memories.
6. Do not add unsupported information.
7. Do not overgeneralize one event into a long-term profile.

======================== Dimension Rules ========================
Use "" when there is no clear evidence. Use [] for empty keywords.

Original DimMem fields:
- memory_type: fact, episodic, or profile.
- time: backward-compatible time field. For episodic memories, use the event time if known. For current profile/fact memories, use the valid time or source time if appropriate.
- location: physical place, online platform, organization, home/work space, venue, or activity context.
- reason: cause, motivation, trigger, or background condition.
- purpose: goal, intention, or expected outcome.
- keywords: short retrieval phrases. Include important subjects, objects, people, places, tools, values, numbers, activities, and aliases.

P3 event/state fields:
- event_time: the actual time/date when the event happened, is planned to happen, or repeatedly occurs. This is more important than source_time for temporal questions.
- valid_from: when a current status/preference/fact starts being valid.
- valid_to: when a status/preference/fact stops being valid. Use "" if still current or unknown.
- status: one of current, past, planned, completed, cancelled, preference, habit, unknown.
- is_current: true only when the memory describes the user's current/latest status, preference, possession, plan, or active state.

P3 event/value/relation fields:
- subject: the actor or owner, usually "the user".
- action: normalized verb phrase, e.g. bought, submitted, received, visited, serviced, plans_to_buy, prefers, dislikes, was_preapproved_for.
- object: the main entity or object of the action.
- value: the answer-like value attached to the memory, e.g. "$400,000", "10%", "9:00 AM", "Instant Pot", "aunt".
- quantity: numeric quantity if applicable, e.g. "2", "6", "15".
- unit: unit of value or quantity, e.g. dollars, percent, days, pounds, miles, minutes.
- relation: compact relation triple or key-value relation, e.g. "mortgage_amount=$400,000", "discount=10%", "giver=aunt", "submission_date=February 1st".
- evidence_span: a short exact or near-exact source phrase supporting the memory. Keep it under 30 words.

======================== Important Extraction Patterns ========================

A. Value-bearing memories
If the user mentions a number, amount, discount, date, time, duration, count, or price, attach it to the correct object.
Example:
User text: "I got a 10% discount on my first purchase from that new clothing brand."
Memory:
content: "The user got a 10% discount on the user's first purchase from a new clothing brand."
value: "10%"
unit: "percent"
relation: "discount=10%; purchase=first purchase; brand=new clothing brand"

B. Event-time memories
If an event has a real event time, put that time in event_time.
Do not confuse event_time with the extraction/source time.
Example:
User text timestamp: 2023-05-26.
User text: "I rearranged my living room around May 5."
event_time: "2023-05-05"
time: "2023-05-05"

C. Before/after memories
If the user says one event happened before or after another, preserve both the event and relation.
Example:
"before getting the Air Fryer, I invested in an Instant Pot"
Memory 1:
content: "The user invested in an Instant Pot before getting the Air Fryer."
object: "Instant Pot"
relation: "before=getting the Air Fryer"

D. Current-state/update memories
If the user updates an older fact, extract the new fact as current.
If the text explicitly indicates replacement/change, use status=current and is_current=true for the new fact.
Do not mark old facts as current unless the source says they are still current.

E. Count/order/list memories
For multiple entities in one message, split them when needed.
Example:
"The user serviced the road bike and planned to replace the commuter bike tire."
Extract separate memories for road bike and commuter bike if both may be counted later.

======================== Extraction Rules ========================
1. Process messages in chronological order.
2. Extract mainly from user messages.
3. Assistant messages may be used as context, but do not extract generic assistant advice as the user's memory unless the user adopts, confirms, refers to, or asks about it.
4. Each memory should be atomic.
5. Preserve key details: person, object, action, time, event_time, location, value, quantity, relation, reason, purpose, and preference.
6. Normalize relative time based on message timestamp when available.
7. The output must be valid JSON only.

{overlap_rule}

Here is the real input you need to process:
{conversation}
"""
OVERLAP_RULE = '''
========================
Input and Overlap Context Rules
========================

You will receive a "current conversation segment".
The first {overlap_count} messages in the input, namely messages numbered 1 to {overlap_count}, are overlapping context from the previous segment and must not be used as new memory sources.
The first {overlap_count} messages may only be used to understand later messages; the content that is actually allowed for extraction starts from message {extract_start_index}.
'''

LONGMEMEVAL_QUERY_ANALYSIS_PROMPT = """ You are a memory query parser. Convert natural language questions into structured retrieval queries for DimMem.

Output only valid JSON.

== Output Format ==
{
  "query_anchor": "",
  "need_assistant_context": false,
  "query_type": "lookup",
  "statefulness": "unknown",
  "dimension": {
    "target_memory_type": [],
    "keywords": [],
    "entities": [],
    "aliases": [],
    "time": "",
    "time_range": {
      "operator": "",
      "start": "",
      "end": "",
      "relative_event": ""
    },
    "location": ""
  },
  "answer_dim": "",
  "parse_confidence": 0.0
}

== Backward Compatibility ==
You must still fill the original fields:
- query_anchor
- need_assistant_context
- dimension.target_memory_type
- dimension.keywords
- dimension.time
- dimension.location
- answer_dim

The extra P2 fields improve retrieval and reranking:
- query_type
- statefulness
- dimension.entities
- dimension.aliases
- dimension.time_range
- parse_confidence

== Field Rules ==

1. query_anchor
Rewrite the question into a retrieval-friendly sentence.
Convert I/me/my to "the user".
Preserve quantity, order, comparison, recommendation, current/latest, before/after, and assistant-recall intent.

2. need_assistant_context
Set true when the answer depends on what the assistant previously said, suggested, recommended, explained, provided, or asked.

3. query_type
Choose exactly one:
lookup, count, sum, order, compare, recommend, list, abstain_check.

4. statefulness
Choose exactly one:
current, historical, timeline, unknown.

5. dimension.target_memory_type
Use any of: fact, episodic, profile. Use [] when uncertain.

6. dimension.keywords
Short retrieval phrases: objects, people, tools, projects, places, activities, numbers, named entities.

7. dimension.entities
Canonical entities that must be matched if possible.

8. dimension.aliases
Alternative names/synonyms/paraphrases for entities.

9. dimension.time
Only fill when the question contains a time constraint.
Normalize relative time using Question Date when possible.
If the question asks for the time/date itself, leave this empty and set answer_dim = "time".

10. dimension.time_range
Fill when dimension.time is non-empty or the question has before/after/between/current/latest logic.
operator must be one of:
on, before, after, around, between, relative_to_event, latest, current, "".

For relative_to_event, put the event phrase in relative_event.
Example: "before getting the Air Fryer" ->
"time": "before getting the Air Fryer",
"time_range": {"operator": "relative_to_event", "start": "before", "end": "", "relative_event": "getting the Air Fryer"}

11. dimension.location
Only fill explicit location/platform/scene constraints.
If the question asks for the location itself, leave empty and set answer_dim = "location".

12. answer_dim
Choose one:
content, time, location, reason, purpose, keywords, count, sum, order, recommendation, "".

13. parse_confidence
A number between 0 and 1. Lower when the question is ambiguous.

== Input ==
Question Date: {question_date}
Question: {question}
"""
LONGMEMEVAL_OFFLINE_UPDATE_PROMPT = """

"""

__all__ = [
    "LONGMEMEVAL_OFFLINE_UPDATE_PROMPT",
    "LONGMEMEVAL_QUERY_ANALYSIS_PROMPT",
    "LONGMEMEVAL_STRUCTURED_MEMORY_EXTRACTION_PROMPT",
    "OVERLAP_RULE",
]
