-- Total count of assistant messages sent 2026-08-03 to 2026-08-28 (inclusive),
-- across ALL chats (not just the 5k sample used elsewhere) -- for
-- cross-checking the dashboard's "messages per day" figure.
--
-- Assumptions:
--   * message.deleted_at NOT filtered (counts messages regardless of deletion)
--   * chat.deleted_at NOT filtered (counts messages in archived chats too)

SELECT
    COUNT(*) AS assistant_message_count,
    COUNT(*) / 26.0 AS avg_per_day
FROM public.message m
WHERE m.role = 'assistant'
  AND m.created_at >= '2026-08-03'
  AND m.created_at < '2026-08-29';
