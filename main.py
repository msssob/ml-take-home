import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths to models
amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Load tokenizers and models
tokenizer = AutoTokenizer.from_pretrained(amateur_path)
amateur_model = AutoModelForCausalLM.from_pretrained(amateur_path)
expert_model = AutoModelForCausalLM.from_pretrained(expert_path)

# Set models to evaluation mode
amateur_model.eval()
expert_model.eval()

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
    scores,
    results,
    kFactor = 4,
) {
    for (const result of results) {
        const { first, second, outcome } = result;
        const firstScore = scores[first] ?? 1000;
        const secondScore = scores[second] ?? 1000;

        const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
        const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
        let sa = 0.5;
        if (outcome === 1) {
            sa = 1;
        } else if (outcome === -1) {
            sa = 0;
        }
        scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
        scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
    }
    return scores;
}\n```"""

# Prepare prompt
prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)

def contrastive_generation(amateur_model, expert_model, prompt, max_tokens=50):
    # Tokenize the input prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Set up device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amateur_model.to(device)
    expert_model.to(device)
    print("hi")
    input_ids = input_ids.to(device)

    # Start generation
    generated_tokens = input_ids
    for _ in range(max_tokens):

        # Get the logits from both models
        amateur_logits = amateur_model(generated_tokens).logits[:, -1, :]
        expert_logits = expert_model(generated_tokens).logits[:, -1, :]

        # Softmax to get probabilities
        amateur_probs = torch.nn.functional.softmax(amateur_logits, dim=-1)
        expert_probs = torch.nn.functional.softmax(expert_logits, dim=-1)

        # Compute contrastive probabilities: Prefer tokens with higher expert likelihood but considering amateur
        contrastive_probs = (amateur_probs + expert_probs) / 2

        # Sample next token from the contrastive probabilities
        next_token = torch.multinomial(contrastive_probs, 1)
        generated_tokens = torch.cat((generated_tokens, next_token), dim=-1)


        # Stop if the model generates the end token
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode the generated tokens (make sure it's a tensor)
    generated_text = tokenizer.decode(generated_tokens[0].cpu(), skip_special_tokens=True)
    return generated_text

# Generate response using contrastive decoding
response = contrastive_generation(amateur_model, expert_model, prompt)
print(response)
