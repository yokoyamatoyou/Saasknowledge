from generate_faq import generate_faqs_from_chunks


def generate_faq(kb_name: str, max_tokens: int, num_pairs: int, client=None) -> int:
    """Generate FAQs and refresh the search engine."""
    from knowledge_gpt_app.app import refresh_search_engine

    count = generate_faqs_from_chunks(kb_name, max_tokens, num_pairs, client=client)
    refresh_search_engine(kb_name)
    return count
