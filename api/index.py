from flask import Flask, request, jsonify, Response, stream_with_context, send_file
from dotenv import load_dotenv
import anthropic
import pdfplumber
import os
import glob
import json

load_dotenv()

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

HR_DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hr_docs")
_cached_system_prompt = None


def load_hr_documents():
    texts = []
    pdf_files = glob.glob(os.path.join(HR_DOCS_DIR, "*.pdf"))
    for path in pdf_files:
        filename = os.path.basename(path)
        try:
            with pdfplumber.open(path) as pdf:
                pages_text = [p.extract_text() for p in pdf.pages if p.extract_text()]
                if pages_text:
                    texts.append(f"=== {filename} ===\n" + "\n".join(pages_text))
        except Exception as e:
            print(f"[경고] {filename} 읽기 실패: {e}")
    return texts


def build_system_prompt():
    global _cached_system_prompt
    if _cached_system_prompt:
        return _cached_system_prompt
    hr_docs = load_hr_documents()
    base = (
        "당신은 씨엘모빌리티의 HR 전문 AI 챗봇입니다. "
        "임직원들의 인사, 복리후생, 규정, 휴가 등 HR 관련 질문에 친절하고 정확하게 답변하세요. "
        "사용자가 사용하는 언어로 답변하세요. "
        "답변은 마크다운 형식(볼드, 리스트, 헤더 등)을 적극 활용해 읽기 쉽게 작성하세요."
    )
    if hr_docs:
        _cached_system_prompt = base + "\n\n아래 사내 HR 문서를 우선 참고하세요:\n\n" + "\n\n".join(hr_docs)
    else:
        _cached_system_prompt = base
    return _cached_system_prompt


PUBLIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")


@app.route("/")
def index():
    with open(os.path.join(PUBLIC_DIR, "index.html"), encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/logo.png")
def serve_logo():
    return send_file(os.path.join(PUBLIC_DIR, "logo.png"), mimetype="image/png")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    history = data.get("history", [])  # 브라우저에서 전달받은 대화 기록

    if not user_message:
        return jsonify({"error": "메시지가 비어 있습니다."}), 400

    messages = history + [{"role": "user", "content": user_message}]

    def generate():
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=[{
                "type": "text",
                "text": build_system_prompt(),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'delta': text})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/hr-docs", methods=["GET"])
def hr_docs_list():
    pdf_files = glob.glob(os.path.join(HR_DOCS_DIR, "*.pdf"))
    return jsonify({"files": [os.path.basename(f) for f in pdf_files]})


if __name__ == "__main__":
    os.makedirs(HR_DOCS_DIR, exist_ok=True)
    docs = load_hr_documents()
    print(f"[HR 문서] {len(docs)}개 파일 로드됨.")
    app.run(debug=True, port=5000)
