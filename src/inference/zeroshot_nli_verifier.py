import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ZeroShotNLIVerifier:
    """
    Lightweight zero-shot Natural Language Inference (NLI) verifier.
    Used as Stage 1 pre-evaluator in the RAG verification pipeline to filter
    out clear supported claims in milliseconds without calling heavier LLMs.
    """
    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading zero-shot NLI verifier: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        # Read label mappings from config
        self.id2label = self.model.config.id2label
        self.nli_labels = [self.id2label[i].lower() for i in range(len(self.id2label))]
        
    def verify(self, context: str, claim: str, threshold: float = 0.85) -> tuple[bool, str, float]:
        """
        Verify if the context supports the claim (hypothesis).
        
        Args:
            context: Premise text.
            claim: Hypothesis text (RAG claim or sentence).
            threshold: Confidence threshold to bypass Qwen.
            
        Returns:
            is_supported (bool): True if entailment is predicted with confidence >= threshold.
            pred_label (str): 'supported' | 'neutral' | 'contradicted'
            confidence (float): Probability score of the predicted NLI class.
        """
        inputs = self.tokenizer(context, claim, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            
        pred_idx = probs.index(max(probs))
        confidence = probs[pred_idx]
        
        pred_label_raw = self.nli_labels[pred_idx]
        if 'entail' in pred_label_raw or 'support' in pred_label_raw:
            pred_label = 'supported'
        elif 'contradict' in pred_label_raw or 'refut' in pred_label_raw:
            pred_label = 'contradicted'
        else:
            pred_label = 'neutral'
            
        is_supported = (pred_label == 'supported') and (confidence >= threshold)
        return is_supported, pred_label, confidence
