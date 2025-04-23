from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import transformers
# Load GPT-2 XL (1.5B) model and tokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer

# See the `MDLM` collection page on the hub for list of available models.
tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
model_name = 'kuleshov-group/mdlm-owt'
model = AutoModelForMaskedLM.from_pretrained(model_name)

model.eval()
if torch.cuda.is_available():
    model.to("cuda")

# Add pad token if not already there
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Your article goes here
article = """
(CNN)The search for a comic book artist missing in the Cayman Islands since Thursday is now being called a recovery mission. Norman Lee, an artist for DC and Marvel comics, went missing while snorkeling with his wife off the eastern coast of Grand Cayman, CNN affiliate WCVB reported. Strong currents hindered the search, which lasted until Friday evening, Cayman 27 reported. "It is unlikely that we will make any recovery at this stage," Chief Inspector Brad Ebanks told Cayman 27. Lee, 47, of Weymouth, Massachusetts, was known and for his work on "Wolverine Annual," "Supergirl," "Starman" and other comic book titles. Tributes flooded his Facebook page and Twitter from friends, fans and colleagues who knew him from art school and comic conventions. "I cannot express how shaken I am that I will never get the chance to see that smile again, and it saddens me that this world has lost a wonderful man in Norman Lee. To his wife Jan, and his family and all his friends and fans that loved him, my sincerest condolences," friend and fellow graphic artist Chris Kinniery said on Facebook. "I'm so sorry to hear about Norman Lee's disappearance. My condolences go out to his family. ... He was an amazing talent in the industry and it was always a pleasure to work with him," freelance artist .
"""

# Prompt template
prompt = f"Summarize:\n{article.strip()}\nSummary:"

# Tokenize with padding and attention mask
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
if torch.cuda.is_available():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Generate output
output_ids = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
)

# Decode and print
summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\nGenerated Output:\n", summary[len(prompt):].strip())
