from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from shared import upload_utils
from shared.upload_utils import list_metadata_items, save_user_metadata

BASE_KNOWLEDGE_DIR = upload_utils.BASE_KNOWLEDGE_DIR


def _load_items(kb_name: str) -> List[Dict[str, Any]]:
    """Load stored items for the given knowledge base.

    The returned ``id`` for each item matches the metadata filename without the
    ``.json`` extension. Some upload helpers prefix this filename (e.g.,
    ``metadata_1``), so the ID may include that prefix.
    """
    items = []
    for item_id, meta in list_metadata_items(kb_name):
        text = ""
        chunk_path = meta.get("paths", {}).get("chunk_path")
        if chunk_path and Path(chunk_path).exists():
            try:
                txt = Path(chunk_path).read_text(encoding="utf-8")
                mid = len(txt) // 2
                snippet = txt[max(0, mid - 6) : mid + 6]
                text = snippet.strip()
            except Exception:
                pass
        items.append({"id": item_id, "meta": meta, "snippet": text})
    return items


def display_thumbnail_grid(kb_name: str) -> None:
    """Render a 3x3 thumbnail grid for items in the knowledge base."""
    items = _load_items(kb_name)
    if not items:
        st.info("No uploaded items found.")
        return

    per_page = 9
    page_key = f"thumb_page_{kb_name}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    page = st.session_state[page_key]
    total_pages = (len(items) - 1) // per_page + 1
    start = page * per_page
    end = start + per_page
    st.markdown(f"**Page {page+1}/{total_pages}**")
    nav_prev, nav_next = st.columns(2)
    with nav_prev:
        if st.button("Prev", disabled=page == 0, key=f"prev_{kb_name}"):
            st.session_state[page_key] = max(page - 1, 0)
            st.rerun()
    with nav_next:
        if st.button("Next", disabled=page >= total_pages - 1, key=f"next_{kb_name}"):
            st.session_state[page_key] = min(page + 1, total_pages - 1)
            st.rerun()

    grid_items = items[start:end]
    rows = [grid_items[i : i + 3] for i in range(0, len(grid_items), 3)]
    for row in rows:
        cols = st.columns(3)
        for col, item in zip(cols, row):
            with col:
                st.button(
                    item.get("snippet", "(no text)"),
                    key=f"sel_{item['id']}",
                    on_click=lambda iid=item["id"]: st.session_state.update(
                        {"current_editing_id": iid}
                    ),
                )

    edit_id = st.session_state.get("current_editing_id")
    if edit_id:
        target = next((it for it in items if it["id"] == edit_id), None)
        if target:
            st.markdown("---")
            st.subheader(f"Edit metadata for {edit_id}")
            title = st.text_input("Title", value=target["meta"].get("title", ""))
            tags = st.text_input(
                "Tags (comma separated)", value=",".join(target["meta"].get("tags", []))
            )
            if st.button("Save", key=f"save_{edit_id}"):
                save_user_metadata(
                    kb_name,
                    edit_id,
                    title,
                    [t.strip() for t in tags.split(",") if t.strip()],
                )
                st.success("Saved metadata")
                st.session_state.current_editing_id = None
                st.rerun()
