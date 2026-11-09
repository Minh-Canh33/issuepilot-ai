
import os

import httpx
import streamlit as st


API_URL = os.getenv(
    "ISSUEPILOT_API_URL",
    "http://127.0.0.1:8000",
)

st.set_page_config(
    page_title="IssuePilot AI",
    page_icon="🔎",
    layout="centered",
)

st.title("🔎 IssuePilot AI")
st.caption("GitHub Issue Retrieval & RAG Assistant")

st.write(
    "Ask a technical question and explore relevant "
    "GitHub issues with an AI-generated investigation."
)

st.divider()

question = st.text_area(
    "Describe your issue",
    placeholder=(
        "How can I investigate SSL certificate "
        "verification errors in requests?"
    ),
    height=120,
)

ask_button = st.button(
    "🔍 Analyze Issue",
    type="primary",
    use_container_width=True,
)

if ask_button:
    question = question.strip()

    if len(question) < 3:
        st.warning("Please enter at least 3 characters.")

    elif len(question) > 2000:
        st.warning("Question must not exceed 2000 characters.")

    else:
        with st.spinner(
            "Searching GitHub issues and generating an answer..."
        ):
            try:
                response = httpx.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=120,
                )

                response.raise_for_status()

                data = response.json()
                answer = data.get("answer", "")

                if not answer:
                    st.error("The API returned an empty answer.")
                else:
                    st.success("Analysis completed")
                    st.subheader("AI Investigation")
                    st.markdown(answer)

            except httpx.ConnectError:
                st.error(
                    "Cannot connect to FastAPI. "
                    "Make sure the backend is running "
                    "on port 8000."
                )

            except httpx.TimeoutException:
                st.error(
                    "The request timed out. Please try again."
                )

            except httpx.HTTPStatusError as error:
                st.error(
                    f"API error: HTTP "
                    f"{error.response.status_code}"
                )

            except (ValueError, KeyError):
                st.error("Invalid API response.")

st.divider()

st.caption(
    "Powered by Sentence Transformers, ChromaDB, "
    "FastAPI, Streamlit, and Gemini. "
    "AI suggestions require independent verification."
)
