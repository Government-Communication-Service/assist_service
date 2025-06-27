import concurrent.futures
import json
import os
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

# --- Load Environment Variables ---
# Check if .env exists before loading
if os.path.exists(".env"):
    load_dotenv()
else:
    st.warning(
        "`.env` file not found. Please create one with your API credentials "
        "(API_URL, AUTH_SECRET_KEY). Using placeholders.",
        icon="⚠️",
    )

PROMPT_INTRO_TO_ASSIST = (
    "I am a Senior Data Scientist at Cabinet Office. "
    "Give me ten different ways Assist can help me "
    "to become more effective and efficient in my role."
)

PROMPT_MCOM = "What is MCOM?"


PROMPT_GOV_UK_SEARCH_1 = (
    "Give me a summary of new announcements made by the government in the past week, related to AI."
)

PROMPT_GOV_UK_SEARCH_2 = (
    "Search for some recent AI announcements from this week, "
    "and then use the Evaluation Cycle from the central guidance "
    "to suggest a plan for evaluating the impact of the announcement. "
    "Keep your response brief."
)


# Use Streamlit secrets or .env for sensitive data
LOCAL_API_URL = "http://localhost"  # os.getenv("LOCAL_API_URL", "http://localhost") # Example default
DEV_API_URL = os.getenv("DEV_API_BASE_URL", "dev-api-url")
TEST_API_URL = os.getenv("TEST_API_BASE_URL", "test-api-url")
PORT = os.getenv("PORT", 5312)
FULL_API_URL = f"{LOCAL_API_URL}:{PORT}"

# Define API URL options for dropdown
API_URL_OPTIONS = {
    "Local": FULL_API_URL,
    "Development": DEV_API_URL,
    "Test": TEST_API_URL,
}

DEFAULT_USER_KEY_UUID = os.getenv("DEFAULT_USER_KEY_UUID", "your-user-key-uuid")
DEFAULT_AUTH_TOKEN = os.getenv("AUTH_SECRET_KEY", "your-auth-token")  # Use a more generic name
st.set_page_config(layout="wide")
st.title("API Load Tester 🧪")

# --- Session State Initialization ---
ss = st.session_state
if "results" not in ss:
    ss["results"] = []
if "test_running" not in ss:
    ss["test_running"] = False
if "progress" not in ss:
    ss["progress"] = 0
if "total_requests_sent" not in ss:
    ss["total_requests_sent"] = 0
if "successful_requests" not in ss:
    ss["successful_requests"] = 0
if "failed_requests" not in ss:
    ss["failed_requests"] = 0


# --- Configuration ---
st.sidebar.header("Configuration")
selected_api_env = st.sidebar.selectbox("API Environment", options=list(API_URL_OPTIONS.keys()), index=0)
api_url_base = API_URL_OPTIONS[selected_api_env]
st.sidebar.text(f"Selected URL: {api_url_base}")
user_key_uuid = st.sidebar.text_input("User Key UUID", value=DEFAULT_USER_KEY_UUID, type="password")
auth_token = st.sidebar.text_input("Auth Token", value=DEFAULT_AUTH_TOKEN, type="password")

# --- Request Parameters ---
st.sidebar.subheader("Request Parameters")
prompt = st.sidebar.text_area("Request Prompt/Payload", height=150, value=PROMPT_MCOM)
# Adapt these based on the target API endpoint structure
use_rag = st.sidebar.checkbox("Use RAG", value=True)
use_gov_uk_search = st.sidebar.checkbox("Use GOV.UK Search API", value=False)

# --- Load Test Settings ---
st.sidebar.subheader("Load Test Settings")
num_concurrent_users = st.sidebar.slider("Concurrent Users", min_value=1, max_value=100, value=15)
total_requests = st.sidebar.number_input("Total Requests", min_value=1, max_value=10000, value=45)

# --- Helper Functions ---


def get_auth_session(api_url: str, user_key: str, token: str) -> str | None:
    """Fetches the Session-Auth token."""
    # Adapt this URL if your auth endpoint is different
    auth_session_url = f"{api_url}/v1/auth-sessions"
    headers = {"Auth-Token": token, "User-Key-UUID": user_key}
    try:
        response = requests.post(url=auth_session_url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json().get("Session-Auth")
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting auth session: {e}", icon="🚨")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        st.error(f"Error parsing auth session response: {e}", icon="🚨")
        st.error(f"Raw auth response: {response.text if 'response' in locals() else 'No response object'}")
        return None


def run_single_request(session_auth: str, api_url: str, user_key: str, token: str, payload: dict) -> dict:
    """Runs a single API request and returns timing and status."""
    # Adapt this URL to your target streaming chat endpoint
    chat_url = f"{api_url}/v1/chats/users/{user_key}/stream"
    headers = {
        "Session-Auth": session_auth,
        "User-Key-UUID": user_key,
        "Auth-Token": token,
        "Content-Type": "application/json",  # Ensure content type is set
        "Accept": "application/json",  # Often needed for streaming APIs
    }
    start_time = time.monotonic()
    first_token_time = None
    response_content = ""
    raw_chunks_len = 0  # Store length of raw chunks
    raw_response_last_chunk = ""  # Store the last raw chunk for debugging
    all_raw_chunks = []  # Store all chunks for debugging

    result = {
        "start_time": start_time,
        "ttft": None,
        "total_time": None,
        "success": False,
        "status_code": None,
        "error": None,
        "response_length": 0,  # Based on extracted content length
        "raw_response_length": 0,  # Based on raw chunk length
        "raw_response_last_chunk": "",  # For debugging
    }

    try:
        # Use a session for potential connection pooling benefits
        with requests.Session() as s:
            with s.post(url=chat_url, headers=headers, json=payload, stream=True, timeout=60) as response:
                result["status_code"] = response.status_code
                response.raise_for_status()  # Check for HTTP errors immediately after getting status

                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:  # Ensure chunk is not empty
                        raw_chunks_len += len(chunk.encode("utf-8"))  # Store raw chunk length (bytes)
                        all_raw_chunks.append(chunk)
                        raw_response_last_chunk = chunk  # Keep updating with latest chunk
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                            result["ttft"] = first_token_time - start_time

                        # --- Content Extraction Logic (Needs Adaptation) ---
                        # This part is specific to your API's streaming format.
                        # The example code parsed JSON chunks like: {"message_streamed": {"content": "..."}}
                        # Adapt this try-except block if your format differs.
                        try:
                            parsed_chunk = json.loads(chunk)
                            # Example: Accessing content based on the provided example structure
                            content = parsed_chunk.get("message_streamed", {}).get("content", "")
                            if content:
                                response_content = content
                        except json.JSONDecodeError:
                            # If chunks are not JSON or have a different structure,
                            # you might need to adjust how `response_content` is built.
                            # For simple text streams, you might just append the raw chunk:
                            # response_content += chunk
                            pass  # Decide how to handle non-JSON chunks or different formats
                        # --- End Content Extraction Logic ---

                # Check if we got meaningful content - if not, mark as failure
                if response_content.strip():  # Only consider success if we have non-empty content
                    result["success"] = True
                else:
                    result["success"] = False
                    result["error"] = "Empty response content received"

    except requests.exceptions.Timeout:
        result["error"] = "Request timed out"
    except requests.exceptions.HTTPError as e:
        result["error"] = f"HTTP Error: {e}"
        # Status code is already set before raise_for_status
    except requests.exceptions.RequestException as e:
        result["error"] = f"Request Error: {str(e)}"
        # Attempt to get status code if response exists
        if hasattr(e, "response") and e.response is not None:
            result["status_code"] = e.response.status_code
    except Exception as e:  # Catch any other unexpected errors
        result["error"] = f"Unexpected error: {str(e)}"
    finally:
        end_time = time.monotonic()
        result["total_time"] = end_time - start_time
        # Calculate length based on the extracted content string
        result["response_length"] = len(response_content.encode("utf-8"))
        # Store the total length of raw bytes received
        result["raw_response_length"] = raw_chunks_len
        # Store the last raw chunk for debugging (truncate if too long)
        if len(raw_response_last_chunk) > 500:
            result["raw_response_last_chunk"] = raw_response_last_chunk[-500:]
        else:
            result["raw_response_last_chunk"] = raw_response_last_chunk
        # <<< Add the actual response content to the result dict >>>
        result["response_content"] = response_content

    return result


# --- Test Execution Logic ---
if st.sidebar.button("Start Load Test", type="primary", disabled=ss.test_running):
    # --- Validate Inputs ---
    if not all([api_url_base, user_key_uuid, auth_token]):
        st.sidebar.error("Please provide API URL, User Key UUID, and Auth Token.", icon="⚠️")
        st.stop()

    # --- Reset State ---
    ss["results"] = []
    ss["test_running"] = True
    ss["progress"] = 0
    ss["total_requests_sent"] = 0
    ss["successful_requests"] = 0
    ss["failed_requests"] = 0

    st.info(
        f"Starting load test with {num_concurrent_users} concurrent users for {total_requests} total requests...",
        icon="🚀",
    )

    # --- Get Auth Session Once ---
    session_auth = get_auth_session(api_url_base, user_key_uuid, auth_token)
    if not session_auth:
        st.error("Failed to obtain auth session. Aborting test.", icon="🛑")
        ss["test_running"] = False
        st.stop()  # Stop script execution

    # --- Prepare Placeholders ---
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()
    col1, col2, col3 = st.columns(3)
    metric_success = col1.metric("Successful", 0)
    metric_failed = col2.metric("Failed", 0)
    metric_rps = col3.metric("Requests/Sec (Avg)", "0.0")

    start_run_time = time.monotonic()

    # --- Run Requests Concurrently ---
    results_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_users) as executor:
        futures = []
        # Submit all tasks
        for _ in range(total_requests):
            # Adapt payload structure according to your API's requirements
            payload = {
                "query": prompt,
                "use_case_id": "",  # Example field, adjust or remove
                "use_rag": use_rag,
                "use_gov_uk_search_api": use_gov_uk_search,  # Example field, adjust or remove
            }
            futures.append(
                executor.submit(run_single_request, session_auth, api_url_base, user_key_uuid, auth_token, payload)
            )
            ss["total_requests_sent"] += 1  # Count submitted tasks

        # --- Process Results as they complete ---
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            if not ss["test_running"]:  # Check if stop button was pressed during processing
                break  # Exit the result processing loop

            try:
                result = future.result()
                results_list.append(result)
                completed_count += 1

                if result["success"]:
                    ss["successful_requests"] += 1
                else:
                    ss["failed_requests"] += 1

                # --- Update Real-time Metrics ---
                current_progress_completed = min(int((completed_count / total_requests) * 100), 100)
                progress_text = f"Processing: {completed_count}/{total_requests}"
                progress_bar.progress(current_progress_completed / 100.0, text=progress_text)

                run_duration = time.monotonic() - start_run_time
                avg_rps = completed_count / run_duration if run_duration > 0 else 0

                status_text.info(
                    f"Completed: {completed_count}/{total_requests} | "
                    f"Success: {ss['successful_requests']} | "
                    f"Failed: {ss['failed_requests']}",
                    icon="📊",
                )
                metric_success.metric("Successful", ss["successful_requests"])
                metric_failed.metric(
                    "Failed",
                    ss["failed_requests"],
                    delta=(
                        f"{result.get('status_code', 'N/A')}: {result.get('error', '')[:50]}..."
                        if result.get("error")
                        else None
                    ),
                    delta_color="inverse" if result.get("error") else "normal",
                )
                metric_rps.metric("Requests/Sec (Avg)", f"{avg_rps:.1f}")

            except concurrent.futures.CancelledError:
                # This might happen if stop is pressed very quickly, or thread issues
                st.warning("A request task was cancelled unexpectedly.", icon="⚠️")
                completed_count += 1  # Count it towards completion
                ss["failed_requests"] += 1  # Treat cancelled as failed
            except Exception as e:
                # Catch errors from future.result() itself
                st.error(f"Error retrieving result from a task: {e}", icon="🚨")
                completed_count += 1
                ss["failed_requests"] += 1

    # --- Finalize Test State ---
    ss["test_running"] = False
    ss["results"] = results_list  # Store collected results
    final_message = f"Load test finished. Processed {completed_count} results."
    if completed_count < total_requests and any(f.cancelled() for f in futures):
        final_message += f" Test was stopped early; {total_requests - completed_count} requests may not have completed."
        st.warning(final_message, icon="⚠️")
        progress_bar.progress(completed_count / total_requests, text="Stopped")

    elif completed_count < total_requests:
        final_message += (
            f" NOTE: Only {completed_count} out of {total_requests} requests were processed. Check for errors."
        )
        st.warning(final_message, icon="⚠️")
        progress_bar.progress(completed_count / total_requests, text="Finished (Incomplete)")
    else:
        st.success(final_message, icon="✅")
        progress_bar.progress(1.0, text="Finished")

# --- Stop Button ---
if ss.test_running:
    if st.sidebar.button("Stop Test", type="secondary"):
        ss["test_running"] = False  # Signal threads to stop submitting/processing
        st.warning("Stop request received. Test will halt after processing currently active requests...", icon="🛑")
        # Note: This doesn't forcefully kill threads, just stops the loops.


# --- Display Results ---
st.divider()
st.header("Results Summary")

# Use results from session state
results_to_display = ss.get("results", [])

if not results_to_display:
    st.info("Run a load test to see results here.")
else:
    results_df = pd.DataFrame(results_to_display)

    # --- Calculate Summary Statistics ---
    successful_results = results_df[results_df["success"]]
    failed_results = results_df[~results_df["success"]]

    st.subheader("Overall Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests Attempted", len(results_df))  # Display actual processed count
    col2.metric("Successful", len(successful_results))
    col3.metric("Failed", len(failed_results))
    error_rate = (len(failed_results) / len(results_df)) * 100 if len(results_df) > 0 else 0
    col4.metric("Error Rate", f"{error_rate:.1f}%")

    # Ensure calculations only happen if there are successful results
    if not successful_results.empty:
        # Calculate effective duration: time from first request start to last request end
        first_request_start = results_df["start_time"].min()
        last_request_end = (results_df["start_time"] + results_df["total_time"]).max()
        effective_duration = last_request_end - first_request_start if len(results_df) > 0 else 0

        # Calculate RPS based on all completed requests over the effective duration
        rps_overall = len(results_df) / effective_duration if effective_duration > 0 else 0

        # Time-based stats from successful requests only
        avg_ttft = successful_results["ttft"].mean() if successful_results["ttft"].notna().any() else None
        median_ttft = successful_results["ttft"].median() if successful_results["ttft"].notna().any() else None
        p95_ttft = successful_results["ttft"].quantile(0.95) if successful_results["ttft"].notna().any() else None

        avg_total_time = successful_results["total_time"].mean()
        median_total_time = successful_results["total_time"].median()
        p95_total_time = successful_results["total_time"].quantile(0.95)

        # --- Display Metrics ---
        col1a, col2a, col3a = st.columns(3)
        col1a.metric("Effective Duration (s)", f"{effective_duration:.2f}")
        col2a.metric("Avg. RPS (Overall)", f"{rps_overall:.2f}")
        col3a.metric("Avg. Response Time (s)", f"{avg_total_time:.3f}")

        st.subheader("Response Time Details (Successful Requests)")
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.metric("Median Total Time (s)", f"{median_total_time:.3f}")
            st.metric("95th Percentile Total Time (s)", f"{p95_total_time:.3f}")
        with col_stats2:
            st.metric("Median TTFT (s)", f"{median_ttft:.3f}" if median_ttft is not None else "N/A")
            st.metric("95th Percentile TTFT (s)", f"{p95_ttft:.3f}" if p95_ttft is not None else "N/A")

        # --- Visualizations ---
        st.subheader("Response Time Distributions (Successful Requests)")
        try:
            fig_total = px.histogram(
                successful_results,
                x="total_time",
                nbins=50,
                title="Total Response Time Distribution",
            )
            fig_total.update_layout(xaxis_title="Total Response Time (s)", yaxis_title="Count")
            st.plotly_chart(fig_total, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not plot Total Response Time histogram: {e}")

        # Only plot TTFT if data exists and is plottable
        if successful_results["ttft"].notna().any():
            try:
                fig_ttft = px.histogram(
                    successful_results.dropna(subset=["ttft"]),
                    x="ttft",
                    nbins=50,
                    title="Time to First Token Distribution",
                )
                fig_ttft.update_layout(xaxis_title="Time to First Token (s)", yaxis_title="Count")
                st.plotly_chart(fig_ttft, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot TTFT histogram: {e}")

        else:
            st.info(
                "TTFT data not available for plotting "
                "(perhaps a non-streaming API or no successful requests with TTFT)."
            )

    else:
        # Handle case where there were requests, but none were successful
        if not results_df.empty:
            st.warning("No successful requests completed to calculate detailed timing statistics.", icon="⚠️")
        # If results_df is also empty, the initial message "Run a load test..." is shown

    # --- Error Summary ---
    if not failed_results.empty:
        st.subheader("Error Summary")
        # Group by status code and the first 100 chars of the error message for better grouping
        failed_results["error_short"] = failed_results["error"].astype(str).str[:100]
        error_counts = (
            failed_results.groupby(["status_code", "error_short"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        st.dataframe(error_counts[["status_code", "error_short", "count"]], use_container_width=True)

    # --- Raw Results Table ---
    st.subheader("Raw Results Data")
    with st.expander("View Raw Data Table"):
        # Format columns for better readability
        results_df_display = results_df.copy()
        # Format timing columns to fixed decimals
        for col in ["ttft", "total_time", "start_time"]:
            if col in results_df_display.columns:
                results_df_display[col] = pd.to_numeric(results_df_display[col], errors="coerce").map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else None
                )
        # Convert boolean 'success' to tick/cross
        results_df_display["success"] = results_df_display["success"].map(lambda x: "✅" if x else "❌")
        # Select and reorder columns for display - prioritize debugging fields
        display_cols = [
            "success",
            "status_code",
            "error",
            "response_content",
            "raw_response_last_chunk",
            "total_time",
            "ttft",
            "response_length",
            "raw_response_length",
            "start_time",
        ]
        # Filter out columns that might not exist if no requests ran
        display_cols = [col for col in display_cols if col in results_df_display.columns]
        st.dataframe(results_df_display[display_cols], use_container_width=True)
