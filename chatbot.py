import json
import os
from dotenv import load_dotenv

load_dotenv()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from order_state import Order, OrderItem

# Load static menu data
with open('menu.json', 'r') as f:
    MENU = json.load(f)

# Global order object
current_order = Order()

MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Fix for attention mask warning
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, device

def get_intent(user_input: str) -> str:
    """Very simple keyword‑based intent recognizer.
    Returns one of: 'menu', 'order', 'recommend', 'confirm', 'timing', 'fallback'.
    """
    lowered = user_input.lower()
    if any(word in lowered for word in ['price', 'cost', 'ingredients', 'allergen', 'menu', 'what do you have']):
        return 'menu'
    if any(word in lowered for word in ['confirm', 'is that correct', 'finalize']):
        return 'confirm'
    if any(word in lowered for word in ['i want', 'order', 'get', 'give me', 'take my order']):
        return 'order'
    if any(word in lowered for word in ['recommend', 'suggest', 'what should i have']):
        return 'recommend'
    if any(word in lowered for word in ['pickup', 'delivery', 'time', 'when']):
        return 'timing'
    return 'fallback'

def handle_menu_query(user_input: str) -> str:
    lowered = user_input.lower()
    found_items = []
    for pizza in MENU.get('pizzas', []):
        if pizza['name'].lower() in lowered:
            for size in pizza['sizes']:
                if size in lowered:
                    price = pizza['sizes'][size]
                    return f"A {size} {pizza['name']} costs ${price:.2f}."
            # If name matches but no size, return medium price
            price = pizza['sizes'].get('medium')
            return f"A medium {pizza['name']} costs ${price:.2f}."
        
        # Collect info for general menu listing
        price = pizza['sizes'].get('medium')
        found_items.append(f"{pizza['name']} (${price:.2f})")
    
    # If we get here, no specific pizza was asked for (or at least matched).
    # Since the intent was 'menu', we should list what we have.
    items_str = ", ".join(found_items)
    return f"We have the following pizzas: {items_str}. Which one would you like to know more about?"

def handle_order_intent(user_input: str) -> str:
    lowered = user_input.lower()
    chosen = None
    for pizza in MENU.get('pizzas', []):
        if pizza['name'].lower() in lowered:
            chosen = pizza
            break
    if not chosen:
        return "I couldn't identify which pizza you'd like. Could you specify the name?"
    size = 'medium'
    for s in chosen['sizes']:
        if s in lowered:
            size = s
            break
    crust = ''
    for c in chosen['crusts']:
        if c in lowered:
            crust = c
            break
    toppings = []
    if 'with' in lowered:
        after = lowered.split('with', 1)[1]
        parts = [p.strip() for p in after.replace('and', ',').split(',')]
        toppings = [p for p in parts if p]
    item = OrderItem(category='pizza', name=chosen['name'], size=size, crust=crust, toppings=toppings)
    current_order.add_item(item)
    return f"Got it! Added a {size} {chosen['name']} pizza{' with ' + ', '.join(toppings) if toppings else ''} to your order. Anything else?"

def handle_recommendation(user_input: str) -> str:
    pizza = MENU.get('pizzas', [])[0]
    return f"If you like classic flavors, I recommend our {pizza['name']} pizza. It's a crowd favorite!"

def handle_confirmation(user_input: str) -> str:
    summary = current_order.summary()
    return f"Here is your order summary:\n{summary}\nWould you like to confirm?"

def handle_timing(user_input: str) -> str:
    policies = MENU.get('policies', {})
    pickup = policies.get('pickup_time', 'unknown')
    delivery = policies.get('delivery_time', 'unknown')
    return f"Pickup usually takes {pickup}, and delivery takes about {delivery}."

def chat_loop(tokenizer, model, device):
    print(f"Chatbot ready! Type 'exit' to quit. (Running on {device})")
    chat_history_ids = None
    step = 0
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        
        intent = get_intent(user_input)
        bot_reply = None
        
        # Handle specific intents
        if intent == 'menu':
            bot_reply = handle_menu_query(user_input)
        elif intent == 'order':
            bot_reply = handle_order_intent(user_input)
        elif intent == 'recommend':
            bot_reply = handle_recommendation(user_input)
        elif intent == 'confirm':
            bot_reply = handle_confirmation(user_input)
        elif intent == 'timing':
            bot_reply = handle_timing(user_input)
        
        # If we have a structured reply, print it and update history manually
        if bot_reply:
            print(f"Bot: {bot_reply}")
            # Encode user input + bot reply and append to history
            # We construct the conversation turn: "User input" + EOS + "Bot reply" + EOS
            new_ids = tokenizer.encode(user_input + tokenizer.eos_token + bot_reply + tokenizer.eos_token, return_tensors="pt").to(device)
            if chat_history_ids is not None:
                chat_history_ids = torch.cat([chat_history_ids, new_ids], dim=-1)
            else:
                chat_history_ids = new_ids
        
        else:
            # Fallback to LLM generation
            # Encode the new user input
            new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
            
            # Append tokens to chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
            
            # Generate a response
            # We create an attention mask (1 for real tokens)
            attention_mask = torch.ones(bot_input_ids.shape, device=device)
            
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=2000,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask, # Fix warning
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )
            
            # Decode the valid generated/response part
            # chat_history_ids contains [context... user_input + eos + generated_response]
            # We want to extract just the newly generated response
            response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
            bot_reply = tokenizer.decode(response_ids[0], skip_special_tokens=True)
            
            # If the model didn't generate anything (empty), provide a default
            if not bot_reply.strip():
                bot_reply = "I'm not sure how to respond to that."
            
            print(f"Bot: {bot_reply}")

if __name__ == "__main__":
    tokenizer, model, device = load_model()
    chat_loop(tokenizer, model, device)
