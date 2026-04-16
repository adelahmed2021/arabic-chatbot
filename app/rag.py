from app.config import VALID_SUBJECTS, SUBJECT_CHUNK_CONFIG, TOP_K_BY_SUBJECT
from app.embeddings import co, get_query_embedding
from app.crud import search_similar_chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()



def chunk_text_by_subject(text: str, subject: str):
    cfg = SUBJECT_CHUNK_CONFIG.get[subject]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["max_chars"],
        chunk_overlap=cfg["overlap"],
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_text(text)

def cohere_chat_text(prompt: str, temperature: float = 0.0) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return ""

    response = co.chat(
        model="command-a-03-2025",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
    )

    parts = response.message.content
    texts = [part.text for part in parts if getattr(part, "type", None) == "text"]
    return "\n".join(texts).strip()


def classify_question_llm(question: str) -> str:
    question = (question or "").strip()
    if not question:
        return "nahw"

    subjects_text = ", ".join(VALID_SUBJECTS)

    prompt = f"""
أنت مصنف أسئلة لمادة اللغة العربية للصف الثالث الثانوي.
أعطِ تصنيفًا واحدًا فقط من الفئات التالية:
{subjects_text}

- balagha: سؤال عن الصور الفنية والمحسنات والقيم البلاغية.
- eirab: سؤال عن إعراب كلمة أو جملة أو المحل الإعرابي.
- nahw: سؤال عن قاعدة نحوية عامة (مبتدأ، خبر، نواسخ، اسم التفضيل، التوابع...).
- qeraah: سؤال فهم في دروس القراءة.
- adab_nosoos: سؤال عن المدارس الأدبية أو النصوص.
- taabir: سؤال عن كتابة موضوع أو برقية أو تقرير.
- qessa: سؤال عن قصة الأيام لطه حسين، مثل أسئلة الصبي في طفولته.
- other: غير ذلك.

أرجِع اسم فئة واحد فقط بدون شرح.

السؤال:
{question}
""".strip()

    answer = cohere_chat_text(prompt, temperature=0.0).lower().strip()

    if answer not in VALID_SUBJECTS:
        if "اعراب" in question or "أعرب" in question:
            return "eirab"
        if "بلاغ" in question or "استعارة" in question or "كناية" in question:
            return "balagha"
        if "فاعل" in question or "مفعول" in question or "نحو" in question:
            return "nahw"
        if "قصة" in question or "الأيام" in question or "الصبي" in question:
            return "qessa"
        if "تعبير" in question:
            return "taabir"
        if "قراءة" in question:
            return "qeraah"
        return "adab_nosoos"

    return answer


def build_answer_prompt(subject: str, question: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n".join([chunk.strip() for chunk in retrieved_chunks if chunk and chunk.strip()])

    base = f"""
أنت مدرس لغة عربية للصف الثالث الثانوي في مدرسة حكومية (وزارة التربية والتعليم المصرية).
أجب عن سؤال الطالب بالعربية الفصحى المبسطة.
إن وُجد سياق، فاعتمد عليه في الإجابة، وإن لم يوجد فاعتمد على القواعد الصحيحة للمادة.

السياق:
{context}

السؤال:
{question}
""".strip()

    if subject == "eirab":
        base += (
            "\n\nالقسم الحالي هو (الإعراب):\n"
            "- إن طلب الطالب إعراب كلمة داخل جملة، فأعرب هذه الكلمة فقط.\n"
            "- حلّل الجملة كاملة داخليًا (فعل، فاعل، مفعول به، مبتدأ، خبر، جملة اسمية داخلية...).\n"
            "- في تراكيب مثل: (الطالب اجتهاده مرتفع)، يُفضَّل في الإعراب المدرسي ما يلي:\n"
            "  * الطالب: مبتدأ أول.\n"
            "  * اجتهاده: مبتدأ ثانٍ مرفوع، وهو مضاف، والضمير مضاف إليه.\n"
            "  * مرتفع: خبر المبتدأ الثاني.\n"
            "  * والجملة من المبتدأ الثاني وخبره في محل رفع خبر للمبتدأ الأول.\n"
            "- لا تجعل الكلمة الثانية بعد المبتدأ خبرًا دائمًا؛ فكثيرًا ما تكون مبتدأ ثانيًا، خاصة إذا كانت مضافة لضمير يعود على المبتدأ الأول مثل: اجتهاده، خلقه، مستواه.\n"
            "- إذا كانت الكلمة مضافة إلى ضمير، فغالبًا تعرب مبتدأ ثانيًا، وخبرها يأتي بعدها.\n"
            "- اكتب الإعراب في نقاط قصيرة بالشكل:\n"
            "  - الكلمة: ...\n"
            "  - نوعها: ...\n"
            "  - المحل الإعرابي / الوظيفة: مبتدأ ثانٍ / خبر / مفعول به / ...\n"
            "  - علامة الإعراب: ...\n"
            "  - سبب الإعراب: مع ذكر السبب النحوي الصحيح.\n"
        )
    elif subject == "balagha":
        base += (
            "\n\nالقسم الحالي هو (البلاغة):\n"
            "- استخرج المطلوب: صورة أو محسّن أو أسلوب.\n"
            "- اذكر نوع الصورة وسر جمالها، أو نوع المحسن وغرضه.\n"
        )
    elif subject == "nahw":
        base += (
            "\n\nالقسم الحالي هو (النحو):\n"
            "- وضّح القاعدة بإيجاز شديد، ثم طبّقها على المثال.\n"
        )
    elif subject == "qessa":
        base += (
            "\n\nالقسم الحالي هو (قصة الأيام):\n"
            "- اعتمد على أحداث القصة وشخصياتها كما هي في المنهج.\n"
        )

    if subject == "eirab":
        base += (
            "\nاكتب الآن الإعراب المطلوب فقط، بالنقاط، بدون أي شروحات إضافية أو أسئلة أخرى."
        )
    else:
        base += (
            "\nأجب الآن عن سؤال الطالب فقط إجابة مباشرة حسب المادة، بدون إعراب أو شروحات زائدة."
        )

    return base


def answer_with_rag(db, question: str):
    question = (question or "").strip()
    if not question:
        return {
            "subject": "nahw",
            "answer": "السؤال فارغ.",
            "chunks_used": 0
        }

    subject = classify_question_llm(question)
    query_embedding = get_query_embedding(question)

    top_k = TOP_K_BY_SUBJECT.get(subject, 3)
    retrieved_chunks = search_similar_chunks(
        db=db,
        subject=subject,
        embedding=query_embedding,
        limit=top_k
    )

    clean_chunks = [chunk.strip() for chunk in retrieved_chunks if chunk and chunk.strip()]

    if not clean_chunks:
        return {
            "subject": subject,
            "answer": "لا توجد بيانات كافية لهذه المادة حتى الآن.",
            "chunks_used": 0
        }

    prompt = build_answer_prompt(subject, question, clean_chunks)
    answer = cohere_chat_text(prompt, temperature=0.2)

    return {
        "subject": subject,
        "answer": answer if answer else "تعذر توليد إجابة من الموديل.",
        "chunks_used": len(clean_chunks)
    }