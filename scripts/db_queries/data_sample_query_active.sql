-- Random sample of 5000 chats *active* in 2026-08-03 to 2026-08-28
-- (inclusive) -- i.e. chats with at least one message created in that
-- window, regardless of when the chat itself was created.
-- One row per message, joined to llm (model name) and aggregated
-- chat_document_mapping/document counts (central vs non-central).
--
-- Assumptions:
--   * chat.deleted_at IS NULL only (archived chats excluded)
--   * message rows included regardless of deleted_at (kept in output for
--     later filtering)
--   * chat_document_mapping.deleted_at IS NULL only (currently-linked docs)
--   * document.deleted_at NOT filtered
--   * document counts are per-chat, duplicated across every message row in
--     that chat
--   * once a chat is sampled, ALL of its messages are included (any time),
--     not just the ones within the window

WITH sampled_chats AS (
    SELECT
        c.id,
        c.user_id,
        c.created_at AS chat_created_at,
        c.title,
        c.from_open_chat,
        c.use_rag,
        c.use_gov_uk_search_api,
        c.use_smart_targets
    FROM public.chat c
    WHERE c.deleted_at IS NULL
      AND EXISTS (
          SELECT 1
          FROM public.message m
          WHERE m.chat_id = c.id
            AND m.created_at >= '2026-09-11'
            AND m.created_at < '2026-09-17'
      )
    ORDER BY random()
    LIMIT 2000
),
doc_counts AS (
    SELECT
        cdm.chat_id,
        COUNT(*) FILTER (WHERE d.is_central) AS central_document_count,
        COUNT(*) FILTER (WHERE NOT d.is_central) AS non_central_document_count
    FROM public.chat_document_mapping cdm
    JOIN public.document d ON d.uuid = cdm.document_uuid
    WHERE cdm.deleted_at IS NULL
      AND cdm.chat_id IN (SELECT id FROM sampled_chats)
    GROUP BY cdm.chat_id
)
SELECT
    sc.id AS chat_id,
    sc.user_id,
    sc.chat_created_at,
    sc.title,
    sc.from_open_chat,
    sc.use_rag,
    sc.use_gov_uk_search_api,
    sc.use_smart_targets,
    m.id AS message_id,
    m.created_at AS message_created_at,
    m.role,
    m.tokens,
    m.cache_write_tokens,
    m.cache_read_tokens,
    m.interrupted,
    m.completion_cost,
    m.deleted_at AS message_deleted_at,
    length(m.content) AS content_length,
    length(m.content_enhanced_with_rag) AS content_enhanced_with_rag_length,
    length(m.summary) AS summary_length,
    length(m.citation) AS citation_length,
    length(m.sources) AS sources_length,
    l.model AS llm_model,
    COALESCE(dc.central_document_count, 0) AS central_document_count,
    COALESCE(dc.non_central_document_count, 0) AS non_central_document_count
FROM sampled_chats sc
JOIN public.message m ON m.chat_id = sc.id
LEFT JOIN public.llm l ON l.id = m.llm_id
LEFT JOIN doc_counts dc ON dc.chat_id = sc.id
ORDER BY sc.id, m.created_at;
