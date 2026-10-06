"use strict";
const $ = (id) => document.getElementById(id);
let sessionId = sessionStorage.getItem("raksha.session") || crypto.randomUUID();
let currentDraft = null;
let pendingRequest = null;
let busy = false;
sessionStorage.setItem("raksha.session", sessionId);
let documents;
try { documents = JSON.parse(localStorage.getItem("raksha.documents") || "[]"); }
catch { documents = []; }
if (!Array.isArray(documents)) documents = [];

function status(id, message, error = false) {
  $(id).textContent = message;
  $(id).classList.toggle("error", error);
}
async function request(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  $("raw-response").textContent = JSON.stringify(data, null, 2);
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((item) => item.msg).join("; ") : data.detail;
    throw new Error(data.error?.message || detail || `Request failed (${response.status})`);
  }
  return data;
}
function renderDocuments() {
  $("documents").replaceChildren();
  if (!documents.length) {
    const text = document.createElement("p");
    text.className = "hint"; text.textContent = "No uploads yet.";
    $("documents").append(text);
  }
  for (const doc of documents) {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox"; checkbox.value = doc.document_id;
    const text = document.createElement("span"); text.textContent = doc.filename;
    const small = document.createElement("small");
    small.textContent = `${doc.chunk_count} sections · ${doc.strategy === "fixed" ? "by length" : "by paragraphs"}`;
    text.append(small); label.append(checkbox, text); $("documents").append(label);
  }
}
function addMessage(role, text, sources = []) {
  $("messages").querySelector(".empty")?.remove();
  const message = document.createElement("div"); message.className = "message";
  const label = document.createElement("strong"); label.textContent = role;
  const body = document.createElement("p"); body.textContent = text;
  message.append(label, body);
  if (sources.length) {
    const details = document.createElement("details");
    const summary = document.createElement("summary"); summary.textContent = `Sources (${sources.length})`;
    details.append(summary);
    for (const source of sources) {
      const item = document.createElement("div"); item.className = "source";
      const title = document.createElement("strong");
      title.textContent = `[${source.citation}] ${source.filename}${source.page ? ` · page ${source.page}` : ""}`;
      const content = document.createElement("p"); content.textContent = source.text;
      item.append(title, content); details.append(item);
    }
    message.append(details);
  }
  $("messages").append(message);
  $("messages").scrollTop = $("messages").scrollHeight;
}
function renderBooking(data) {
  currentDraft = data.draft;
  const details = data.booking || data.draft;
  $("booking").hidden = !details;
  $("confirm-booking").hidden = data.booking_status !== "awaiting_confirmation";
  $("booking-details").replaceChildren();
  if (!details) return;
  const fields = data.booking ? ["name", "email", "starts_at", "timezone"] : ["name", "email", "date", "time", "timezone"];
  for (const field of fields) {
    const term = document.createElement("dt"); term.textContent = field === "starts_at" ? "Date & time" : field;
    const value = document.createElement("dd");
    value.textContent = field === "starts_at" ? new Intl.DateTimeFormat("en-GB", { dateStyle: "medium", timeStyle: "short", timeZone: details.timezone }).format(new Date(details[field])) : details[field] || "Not provided";
    $("booking-details").append(term, value);
  }
  $("booking-status").textContent = data.booking_status === "booked" ? "Booking saved." : "Review your details before confirming.";
}
function setBusy(value) {
  busy = value;
  $("chat-form").querySelector("button").disabled = value;
  $("confirm-booking").disabled = value;
  $("new-session").disabled = value;
  $("message").disabled = value;
}
async function sendChat(confirm = false) {
  if (busy) return;
  const message = confirm ? "Confirm booking" : $("message").value.trim();
  if (!message) return;
  const selected = [...$("documents").querySelectorAll("input:checked")].map((input) => input.value);
  const payload = { message, session_id: sessionId, document_ids: selected, confirm_booking: confirm, draft_id: confirm ? currentDraft?.draft_id : null };
  const fingerprint = JSON.stringify(payload);
  if (!pendingRequest || pendingRequest.fingerprint !== fingerprint) {
    pendingRequest = { fingerprint, request_id: crypto.randomUUID() };
    addMessage("You", message);
  }
  payload.request_id = pendingRequest.request_id;
  setBusy(true); status("chat-status", "Processing…");
  try {
    const data = await request("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    addMessage("Assistant", data.answer, data.sources); renderBooking(data);
    sessionId = data.session_id; sessionStorage.setItem("raksha.session", sessionId);
    $("message").value = ""; pendingRequest = null; status("chat-status", "");
  } catch (error) { status("chat-status", error.message, true); }
  finally { setBusy(false); $("message").focus(); }
}
$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button");
  const file = $("file").files[0];
  if (!file) return;
  if (Number($("overlap").value) >= Number($("chunk-size").value)) {
    status("upload-status", "Overlap must be smaller than size.", true); return;
  }
  button.disabled = true; status("upload-status", "Uploading…");
  try {
    const data = await request("/documents", { method: "POST", body: new FormData(event.currentTarget) });
    documents = [data, ...documents.filter((doc) => doc.document_id !== data.document_id)].slice(0, 50);
    localStorage.setItem("raksha.documents", JSON.stringify(documents)); renderDocuments();
    status("upload-status", data.duplicate ? "Already uploaded." : "Ready.");
  } catch (error) { status("upload-status", error.message, true); }
  finally { button.disabled = false; }
});
$("chat-form").addEventListener("submit", (event) => { event.preventDefault(); sendChat(); });
$("confirm-booking").addEventListener("click", () => sendChat(true));
$("new-session").addEventListener("click", () => {
  sessionId = crypto.randomUUID(); sessionStorage.setItem("raksha.session", sessionId);
  currentDraft = null; pendingRequest = null;
  $("messages").replaceChildren(); $("booking").hidden = true;
  $("message").value = ""; status("chat-status", ""); $("message").focus();
});
renderDocuments();
