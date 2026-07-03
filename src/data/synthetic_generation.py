"""
src/data/synthetic_generation.py
=================================
Synthetic Turkish RAG hallucination data generation pipeline.

Pipeline:
  1. Fetch Turkish Wikipedia articles (quality-filtered)
  2. For each article paragraph, generate a question + faithful answer via LLM
  3. Inject controlled errors to create hallucinated answers
  4. Label each (context, claim) pair via LLM → {label, score, error_type, evidence_span, explanation, decision}
  5. Save as JSONL matching the project schema

LLM backend: OpenRouter (OpenAI-compatible API)
Default model: meta-llama/llama-3.3-70b-instruct

Usage (from synthetic_generation import):
    gen = SyntheticDataGenerator()
    examples = gen.generate(n_target=200, domains=["literature", "history"])
    gen.save(examples, "data/synthetic/pilot_v1.jsonl")
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import wikipediaapi
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

load_dotenv()

# Windows cp1254 fix: force utf-8 output, disable Rich spinner (uses Unicode braille chars)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

# Domain → list of Turkish Wikipedia seed article titles
DOMAIN_SEEDS: dict[str, list[str]] = {
    "literature": [
        "Orhan Pamuk", "Yaşar Kemal", "Nazım Hikmet", "Halide Edib Adıvar",
        "Ahmet Hamdi Tanpınar", "Sabahattin Ali", "Reşat Nuri Güntekin",
        "Ömer Seyfettin", "Peyami Safa", "Kemal Tahir",
        "İnce Memed", "Benim Adım Kırmızı", "Türk edebiyatı",
        "Cumhuriyet dönemi Türk edebiyatı", "Divan edebiyatı",
        "Fuzuli", "Baki (şair)", "Nedim (şair)", "Yunus Emre",
        "Tanzimat edebiyatı", "Namık Kemal", "Ziya Paşa",
        "Servet-i Fünun edebiyatı", "Halit Ziya Uşaklıgil",
        "Milli edebiyat dönemi", "Mehmet Akif Ersoy",
    ],
    "history": [
        "Mustafa Kemal Atatürk", "Osmanlı İmparatorluğu", "Kurtuluş Savaşı",
        "Türkiye Cumhuriyeti", "Fatih Sultan Mehmet", "Kanuni Sultan Süleyman",
        "Birinci Dünya Savaşı", "İkinci Dünya Savaşı",
        "Bizans İmparatorluğu", "Osmanlı-Rus Savaşı (1877-1878)",
        "Çanakkale Savaşı", "Türk Kurtuluş Savaşı",
        "Osmanlı padişahları", "Yavuz Sultan Selim",
        "II. Abdülhamit", "Tanzimat", "Meşrutiyet",
        "Cumhuriyet Halk Partisi", "Demokrat Parti",
        "Türkiye'nin NATO üyeliği",
    ],
    "geography": [
        "Türkiye", "İstanbul", "Ankara", "İzmir", "Karadeniz",
        "Ege Denizi", "Anadolu", "Kızılırmak", "Fırat Nehri", "Ağrı Dağı",
        "Marmara Denizi", "Akdeniz", "Boğaziçi", "Çanakkale Boğazı",
        "Kapadokya", "Pamukkale", "Nemrut Dağı",
        "Türkiye'nin coğrafyası", "Doğu Anadolu Bölgesi",
        "Güneydoğu Anadolu Bölgesi", "Ege Bölgesi",
        "Marmara Bölgesi", "İç Anadolu Bölgesi",
    ],
    "science": [
        "DNA", "Evrim teorisi", "Kara delik", "Kuantum mekaniği",
        "Periyodik tablo", "Fotosentez", "Bağışıklık sistemi",
        "Büyük Patlama", "Genel görelilik", "Özel görelilik",
        "Atom", "Elektron", "Proton", "Nötron",
        "Hücre (biyoloji)", "Mitoz", "Mayoz",
        "Yerçekimi", "Elektromanyetizma", "Termodinamik",
        "Newton'un hareket yasaları", "Mendel genetiği",
    ],
    "technology": [
        "Yapay zeka", "Makine öğrenmesi", "İnternet",
        "Linux", "Elektrikli araç",
        "Yenilenebilir enerji", "Güneş enerjisi", "Rüzgar enerjisi",
        "Bilgisayar", "Yazılım", "Donanım",
        "Akıllı telefon", "Bulut bilişim", "Siber güvenlik",
        "Robotik", "Otomasyon", "Yarı iletken",
        "Transistör", "Entegre devre",
    ],
    "education": [
        "Üniversite", "Eğitim", "Öğretmen",
        "Türkiye'deki üniversiteler", "Boğaziçi Üniversitesi",
        "Orta Doğu Teknik Üniversitesi", "İstanbul Teknik Üniversitesi",
        "Ankara Üniversitesi", "Hacettepe Üniversitesi",
        "İlköğretim", "Ortaöğretim", "Yükseköğretim",
        "Pedagoji", "Uzaktan eğitim",
    ],
    "health": [
        "Diyabet", "Hipertansiyon", "Kanser", "Antibiyotik",
        "COVID-19", "İnfluenza", "Tüberküloz",
        "Kalp", "Beyin", "Karaciğer", "Böbrek",
        "Bağışıklık sistemi", "Aşılama",
        "Cerrahi", "Anestezi", "Radyoloji",
        "Psikoloji", "Psikiyatri", "Depresyon",
    ],
    "law": [
        "Türkiye Cumhuriyeti Anayasası", "Avrupa İnsan Hakları Mahkemesi",
        "Hukuk", "Ceza hukuku", "Medeni hukuk",
        "Anayasa Mahkemesi", "Yargıtay", "Danıştay",
        "İnsan hakları", "Temel haklar ve özgürlükler",
        "Uluslararası hukuk", "Avrupa Birliği hukuku",
        "İş hukuku", "Ticaret hukuku",
    ],
}

# Error injection types and their weights
ERROR_TYPES = [
    "entity_error",
    "date_number_error",
    "relation_error",
    "attribution_error",
    "unsupported_inference",
    "negation_error",
    "overgeneralization",
]

# Label distribution targets for generated data
LABEL_DISTRIBUTION = {
    "supported": 0.33,
    "contradicted": 0.20,
    "unsupported": 0.20,
    "partially_supported": 0.14,
    "insufficient_context": 0.13,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GeneratedExample:
    id: str
    question: str
    context: str
    answer: str
    claim: str
    label: str
    hallucination_score: float
    error_type: str
    evidence_span: str
    explanation: str
    decision: str
    source_type: str = "synthetic"
    generator_model: str = ""
    prompt_version: str = "v1.0"
    annotator_id: str = "llm_synth_v1"
    annotation_confidence: float = 0.85
    annotation_round: int = 1
    domain: str = "other"
    claim_index: int = 0
    claim_count_in_answer: int = 1
    context_source_count: int = 1
    context_truncated: bool = False
    evidence_source_index: int = 0
    split: str = "train"
    notes: str = "synthetic pilot data"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "context_source_count": self.context_source_count,
            "context_truncated": self.context_truncated,
            "answer": self.answer,
            "claim": self.claim,
            "claim_index": self.claim_index,
            "claim_count_in_answer": self.claim_count_in_answer,
            "label": self.label,
            "hallucination_score": self.hallucination_score,
            "error_type": self.error_type,
            "evidence_span": self.evidence_span,
            "evidence_source_index": self.evidence_source_index,
            "explanation": self.explanation,
            "decision": self.decision,
            "source_type": self.source_type,
            "generator_model": self.generator_model,
            "prompt_version": self.prompt_version,
            "annotator_id": self.annotator_id,
            "annotation_confidence": self.annotation_confidence,
            "annotation_round": self.annotation_round,
            "domain": self.domain,
            "split": self.split,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Wikipedia fetcher
# ---------------------------------------------------------------------------

class WikipediaFetcher:
    """Fetches and filters Turkish Wikipedia article paragraphs."""

    MIN_WORDS = 80   # minimum words per paragraph (context window)
    MAX_WORDS = 300  # maximum words per paragraph (keep context focused)
    MIN_ARTICLE_WORDS = 500  # minimum total article length

    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            language="tr",
            user_agent="TurkishRAGVerifier/1.0 (research project)",
        )
        self._cache: dict[str, list[str]] = {}

    def fetch_paragraphs(self, title: str) -> list[str]:
        """Return quality-filtered paragraphs from a Turkish Wikipedia article."""
        if title in self._cache:
            return self._cache[title]

        page = self.wiki.page(title)
        if not page.exists():
            console.print(f"[yellow]Wikipedia page not found: {title}[/yellow]")
            return []

        full_text = page.text
        total_words = len(full_text.split())
        if total_words < self.MIN_ARTICLE_WORDS:
            console.print(f"[yellow]Article too short ({total_words} words): {title}[/yellow]")
            return []

        # Split into paragraphs, filter by length
        raw_paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        good = []
        for para in raw_paragraphs:
            words = para.split()
            if self.MIN_WORDS <= len(words) <= self.MAX_WORDS:
                # Skip paragraphs that look like section headers or lists
                if not para.startswith("==") and not para.startswith("*"):
                    good.append(para)

        self._cache[title] = good
        return good

    def get_paragraphs_for_domain(
        self, domain: str, max_per_domain: int = 50
    ) -> list[tuple[str, str]]:
        """Return list of (domain, paragraph) tuples for a domain."""
        seeds = DOMAIN_SEEDS.get(domain, [])
        results: list[tuple[str, str]] = []
        for title in seeds:
            paras = self.fetch_paragraphs(title)
            for p in paras[:5]:  # max 5 paragraphs per article
                results.append((domain, p))
            if len(results) >= max_per_domain:
                break
        return results[:max_per_domain]


# ---------------------------------------------------------------------------
# OpenRouter LLM client
# ---------------------------------------------------------------------------

class OpenRouterClient:
    """Thin wrapper around OpenAI client pointed at OpenRouter."""

    def __init__(self, model: Optional[str] = None):
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
            raise ValueError(
                "OPENROUTER_API_KEY not set. "
                "Edit turkish-rag-verifier/.env and add your key."
            )
        self.model = model or OPENROUTER_MODEL
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> str:
        """Call the LLM and return the assistant message content."""
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                wait = 2 ** attempt
                console.print(f"[red]LLM error (attempt {attempt+1}/{retries}): {e}[/red]")
                if attempt < retries - 1:
                    time.sleep(wait)
        return ""

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> dict:
        """Call LLM and parse JSON response. Returns empty dict on failure."""
        raw = self.chat(system, user, temperature=temperature, max_tokens=max_tokens)
        return _parse_json_safe(raw)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_json_safe(text: str) -> dict:
    """Try to extract JSON from LLM output (handles markdown code blocks)."""
    if not text:
        return {}
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    # Find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Single-call prompt (3x faster: QA + error injection + labeling in one call)
# ---------------------------------------------------------------------------

SINGLE_CALL_SYSTEM = """Sen Türkçe RAG halüsinasyon eğitim verisi üreten bir asistansın.
Verilen bağlam paragrafı ve hedef etiket için tek seferde eksiksiz bir eğitim örneği üret.
Çıktı geçerli JSON olmalıdır."""

SINGLE_CALL_USER = """Aşağıdaki Türkçe Wikipedia paragrafından bir eğitim örneği üret.

Bağlam:
{context}

Hedef etiket: {target_label}
{error_hint}

Görev:
1. Bağlama dayalı bir soru üret
2. Bağlamdan doğru bir cevap üret (faithful_answer)
3. Eğer hedef etiket 'supported' değilse: faithful_answer'a '{error_type}' türünde kontrollü bir hata ekleyerek final_answer üret
4. final_answer'ı bağlama göre değerlendir ve etiketle

Kurallar:
- Soru bağlamdan doğrudan cevaplanabilir olmalı
- faithful_answer 1-2 cümle, yalnızca bağlamdaki bilgilere dayalı
- Hata varsa: bağlamla çelişmeli veya bağlamda bulunmamalı, doğal Türkçe kalmalı
- Etiket gerçekten bağlama göre verilmeli (hedef etiket rehber, ama LLM kararı öncelikli)
- evidence_span: bağlamdan kopyalanan kanıt metni

Çıktı formatı (JSON):
{{
  "question": "...",
  "faithful_answer": "...",
  "final_answer": "...",
  "label": "supported|partially_supported|unsupported|contradicted|insufficient_context",
  "hallucination_score": 0.0,
  "error_type": "entity_error|date_number_error|relation_error|attribution_error|unsupported_inference|fabricated_source|constraint_violation|omission|negation_error|overgeneralization|none",
  "evidence_span": "...",
  "explanation": "...",
  "decision": "accept|warn|revise|reject|web_check|insufficient_context"
}}

Notlar:
- hallucination_score: supported→0.05, partially_supported→0.35, insufficient_context→0.45, unsupported→0.75, contradicted→0.95
- label=supported ise error_type mutlaka "none" olmalı
- label=supported ise final_answer = faithful_answer olmalı"""

ERROR_TYPE_HINTS = {
    "contradicted": "Hata türü önerisi: entity_error veya date_number_error (bağlamla açıkça çelişen bir hata ekle)",
    "unsupported": "Hata türü önerisi: unsupported_inference veya overgeneralization (bağlamda olmayan bir bilgi ekle)",
    "partially_supported": "Hata türü önerisi: overgeneralization veya omission (kısmen desteklenen ama genişletilmiş bir iddia)",
    "insufficient_context": "Bağlam bu konuda karar vermek için yetersiz kalacak şekilde soru üret",
    "supported": "",
}

# Legacy prompts kept for reference (not used in single-call mode)
QA_SYSTEM = ""
QA_USER = ""
ERROR_INJECTION_SYSTEM = ""
ERROR_INJECTION_USER = ""
LABEL_SYSTEM = ""
LABEL_USER = ""
ERROR_DESCRIPTIONS = {}

ERROR_DESCRIPTIONS = {
    "entity_error": "Yanlış kişi, yer veya nesne adı kullan (örn. farklı bir isim, şehir veya eser adı)",
    "date_number_error": "Yanlış tarih, yıl, sayı veya ölçü kullan (örn. yılı değiştir, sayıyı yanlış yaz)",
    "relation_error": "İki doğru bilgi arasında yanlış bir ilişki kur (örn. kim kimi etkiledi, ne neye bağlı)",
    "attribution_error": "Yanlış atıf yap (örn. başka birine atfet, yanlış kişinin yaptığını söyle)",
    "unsupported_inference": "Bağlamdan desteklenmeyen bir çıkarım ekle (bağlamda olmayan ama mantıklı görünen bir bilgi)",
    "negation_error": "Olumsuzlama hatası yap (örn. 'değildir' yerine 'dir' veya tam tersi)",
    "overgeneralization": "Aşırı genelleme yap (örn. 'bazı' yerine 'tüm', 'en önemli' gibi desteklenmeyen iddialar)",
}


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generates synthetic Turkish RAG hallucination examples.

    Example usage:
        gen = SyntheticDataGenerator(model="meta-llama/llama-3.3-70b-instruct")
        examples = gen.generate(n_target=500, domains=["literature", "history"])
        gen.save(examples, "data/synthetic/pilot_v1.jsonl")
    """

    def __init__(self, model: Optional[str] = None):
        self.llm = OpenRouterClient(model=model)
        self.wiki = WikipediaFetcher()
        self.model_name = self.llm.model
        self._id_counter = 1

    def _next_id(self) -> str:
        uid = f"trrag_syn_{self._id_counter:06d}"
        self._id_counter += 1
        return uid

    def _generate_qa(self, context: str) -> tuple[str, str]:
        """Generate a question-answer pair from a context paragraph."""
        result = self.llm.chat_json(
            system=QA_SYSTEM,
            user=QA_USER.format(context=context),
            temperature=0.7,
        )
        question = result.get("question", "").strip()
        answer = result.get("answer", "").strip()
        return question, answer

    def _inject_error(self, context: str, answer: str, error_type: str) -> tuple[str, str]:
        """Inject a controlled error into the answer. Returns (modified_answer, what_changed)."""
        desc = ERROR_DESCRIPTIONS.get(error_type, "Bir hata ekle")
        result = self.llm.chat_json(
            system=ERROR_INJECTION_SYSTEM,
            user=ERROR_INJECTION_USER.format(
                context=context,
                answer=answer,
                error_type=error_type,
                error_description=desc,
            ),
            temperature=0.8,
        )
        modified = result.get("modified_answer", "").strip()
        what_changed = result.get("what_changed", "").strip()
        if not modified:
            return answer, ""
        return modified, what_changed

    def _label_claim(
        self, question: str, context: str, claim: str
    ) -> dict:
        """Label a (question, context, claim) triple via LLM."""
        result = self.llm.chat_json(
            system=LABEL_SYSTEM,
            user=LABEL_USER.format(
                question=question,
                context=context,
                claim=claim,
            ),
            temperature=0.2,
        )
        return result

    def _validate_label_result(self, result: dict) -> bool:
        """Check that LLM returned a valid label dict."""
        valid_labels = {
            "supported", "partially_supported", "unsupported",
            "contradicted", "insufficient_context",
        }
        valid_decisions = {
            "accept", "warn", "revise", "reject",
            "web_check", "insufficient_context",
        }
        valid_error_types = {
            "entity_error", "date_number_error", "relation_error",
            "attribution_error", "unsupported_inference", "fabricated_source",
            "constraint_violation", "omission", "negation_error",
            "overgeneralization", "none",
        }
        if result.get("label") not in valid_labels:
            return False
        score = result.get("hallucination_score", -1)
        if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
            return False
        if result.get("error_type") not in valid_error_types:
            return False
        if result.get("decision") not in valid_decisions:
            return False
        return True

    def _pick_target_label(self, rng: random.Random) -> str:
        """Pick a label according to target distribution."""
        labels = list(LABEL_DISTRIBUTION.keys())
        weights = [LABEL_DISTRIBUTION[l] for l in labels]
        return rng.choices(labels, weights=weights, k=1)[0]

    def generate_one(
        self,
        domain: str,
        context: str,
        target_label: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ) -> Optional[GeneratedExample]:
        """
        Generate a single labeled example from a context paragraph.
        Uses a single LLM call (3x faster than the 3-call approach).
        """
        if rng is None:
            rng = random.Random()

        if target_label is None:
            target_label = self._pick_target_label(rng)

        # Pick error type hint for non-supported labels
        if target_label == "contradicted":
            error_type = rng.choice(["entity_error", "date_number_error", "negation_error"])
        elif target_label == "unsupported":
            error_type = rng.choice(["unsupported_inference", "fabricated_source", "overgeneralization"])
        elif target_label == "partially_supported":
            error_type = rng.choice(["overgeneralization", "unsupported_inference", "omission"])
        else:
            error_type = "none"

        error_hint = ERROR_TYPE_HINTS.get(target_label, "")

        # Single LLM call: QA + error injection + labeling
        result = self.llm.chat_json(
            system=SINGLE_CALL_SYSTEM,
            user=SINGLE_CALL_USER.format(
                context=context,
                target_label=target_label,
                error_hint=error_hint,
                error_type=error_type,
            ),
            temperature=0.7,
            max_tokens=1200,
        )

        if not result or not self._validate_label_result(result):
            return None

        question = result.get("question", "").strip()
        final_answer = result.get("final_answer", "").strip()
        if not question or not final_answer:
            return None

        # Extract claim (first sentence)
        claim = final_answer.split(".")[0].strip()
        if len(claim) < 10:
            claim = final_answer

        label = result["label"]
        score = float(result.get("hallucination_score", 0.5))
        error_type = result.get("error_type", "none")
        evidence_span = result.get("evidence_span", "")
        explanation = result.get("explanation", "")
        decision = result.get("decision", "warn")
        answer_to_label = final_answer

        # Clamp score to valid range per label
        score_ranges = {
            "supported": (0.00, 0.15),
            "partially_supported": (0.20, 0.50),
            "insufficient_context": (0.30, 0.60),
            "unsupported": (0.60, 0.85),
            "contradicted": (0.80, 1.00),
        }
        lo, hi = score_ranges.get(label, (0.0, 1.0))
        score = max(lo, min(hi, score))

        return GeneratedExample(
            id=self._next_id(),
            question=question,
            context=context,
            answer=answer_to_label,
            claim=claim,
            label=label,
            hallucination_score=round(score, 4),
            error_type=error_type,
            evidence_span=evidence_span,
            explanation=explanation,
            decision=decision,
            source_type="synthetic",
            generator_model=self.model_name,
            prompt_version="v1.0",
            annotator_id="llm_synth_v1",
            annotation_confidence=0.85,
            annotation_round=1,
            domain=domain,
            claim_index=0,
            claim_count_in_answer=1,
            context_source_count=1,
            context_truncated=False,
            evidence_source_index=0,
            split="train",  # split assigned later by build_splits.py
            notes="synthetic pilot data",
        )

    def generate(
        self,
        n_target: int = 500,
        domains: Optional[list[str]] = None,
        seed: int = 42,
        max_retries_per_example: int = 2,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 50,
    ) -> list[GeneratedExample]:
        """
        Generate n_target labeled examples across specified domains.

        Args:
            n_target: Total number of examples to generate
            domains: List of domain names (default: all domains)
            seed: Random seed for reproducibility
            max_retries_per_example: How many times to retry a failed example
            checkpoint_path: If set, save intermediate results every checkpoint_every examples
            checkpoint_every: Save checkpoint every N examples (default: 50)

        Returns:
            List of GeneratedExample objects
        """
        rng = random.Random(seed)
        if domains is None:
            domains = list(DOMAIN_SEEDS.keys())

        console.print(f"\n[bold cyan]Synthetic data generation[/bold cyan]")
        console.print(f"  Target: {n_target} examples")
        console.print(f"  Domains: {domains}")
        console.print(f"  Model: {self.model_name}")
        console.print(f"  Seed: {seed}\n")

        # Collect paragraphs from Wikipedia
        console.print("[bold]Step 1: Fetching Wikipedia paragraphs...[/bold]")
        all_paragraphs: list[tuple[str, str]] = []  # (domain, paragraph)
        for domain in domains:
            paras = self.wiki.get_paragraphs_for_domain(domain, max_per_domain=60)
            all_paragraphs.extend(paras)
            console.print(f"  {domain}: {len(paras)} paragraphs")

        if not all_paragraphs:
            console.print("[red]No paragraphs fetched. Check Wikipedia connectivity.[/red]")
            return []

        console.print(f"  Total: {len(all_paragraphs)} paragraphs available\n")

        # Shuffle paragraphs
        rng.shuffle(all_paragraphs)

        # Generate examples
        console.print("[bold]Step 2: Generating labeled examples...[/bold]")
        examples: list[GeneratedExample] = []
        para_idx = 0
        failed = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating...", total=n_target)

            # Each paragraph can be reused up to max_examples_per_para times
            # with different target labels to reach n_target
            max_examples_per_para = max(1, (n_target // max(len(all_paragraphs), 1)) + 2)

            while len(examples) < n_target:
                if para_idx >= len(all_paragraphs) * max_examples_per_para:
                    console.print(
                        f"Ran out of paragraphs after {len(examples)} examples "
                        f"({failed} failed). Consider adding more seed articles."
                    )
                    break

                domain, context = all_paragraphs[para_idx % len(all_paragraphs)]
                para_idx += 1

                # Pick target label
                target_label = self._pick_target_label(rng)

                example = None
                for _ in range(max_retries_per_example):
                    try:
                        example = self.generate_one(
                            domain=domain,
                            context=context,
                            target_label=target_label,
                            rng=rng,
                        )
                        if example:
                            break
                    except Exception as e:
                        console.print(f"Error generating example: {e}")

                if example:
                    examples.append(example)
                    progress.update(task, advance=1)
                    if len(examples) % checkpoint_every == 0:
                        console.print(f"  Progress: {len(examples)}/{n_target} examples generated")
                        if checkpoint_path:
                            self.save(examples, checkpoint_path)
                            console.print(f"  Checkpoint saved: {checkpoint_path}")
                else:
                    failed += 1

        console.print(f"\n[green]Generated {len(examples)} examples ({failed} failed)[/green]")

        # Print label distribution
        from collections import Counter
        label_counts = Counter(e.label for e in examples)
        console.print("\n[bold]Label distribution:[/bold]")
        for label, count in sorted(label_counts.items()):
            pct = count / len(examples) * 100 if examples else 0
            console.print(f"  {label}: {count} ({pct:.1f}%)")

        return examples

    def save(self, examples: list[GeneratedExample], output_path: str) -> Path:
        """Save examples to JSONL file. Returns the output path."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")

        console.print(f"\n[green]Saved {len(examples)} examples to {out}[/green]")
        return out