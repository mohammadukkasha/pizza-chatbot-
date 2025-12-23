import pytest
from chatbot import (
    get_intent,
    handle_menu_query,
    handle_order_intent,
    handle_confirmation,
    handle_timing,
    current_order,
    MENU,
)

def test_menu_query_price():
    query = "What is the price of a large Pepperoni?"
    response = handle_menu_query(query)
    assert "large" in response.lower()
    assert "pepperoni" in response.lower()
    assert "$" in response

def test_order_intent_adds_item():
    # Reset order
    current_order.items.clear()
    order_input = "I want a medium Margherita with basil"
    response = handle_order_intent(order_input)
    assert "added" in response.lower()
    # Verify order content
    assert len(current_order.items) == 1
    item = current_order.items[0]
    assert item.name.lower() == "margherita"
    assert item.size == "medium"
    assert "basil" in item.toppings

def test_confirmation_summary():
    # Ensure there is at least one item
    if not current_order.items:
        current_order.add_item(
            type('Dummy', (), {
                'category': 'pizza',
                'name': 'TestPizza',
                'size': 'small',
                'crust': '',
                'toppings': [],
                'quantity': 1,
                'special': ''
            })()
        )
    summary = handle_confirmation("")
    assert "order summary" in summary.lower()
    assert "pizza" in summary.lower()

def test_timing_info():
    response = handle_timing("")
    assert "pickup" in response.lower()
    assert "delivery" in response.lower()

def test_menu_general_query():
    query = "menu please"
    response = handle_menu_query(query)
    assert "We have the following pizzas" in response
    assert "Margherita" in response
    assert "Pepperoni" in response

# Simple integration test simulating a conversation flow
def test_conversation_flow():
    current_order.items.clear()
    # 1. Ask price
    assert get_intent("What is the price of a small Margherita?") == "menu"
    # 2. Place order
    assert get_intent("I want a small Margherita") == "order"
    handle_order_intent("I want a small Margherita")
    # 3. Confirm
    assert get_intent("Can you confirm my order?") == "confirm"
    confirmation = handle_confirmation("")
    assert "margherita" in confirmation.lower()
    assert "small" in confirmation.lower()
    # 4. Timing
    assert get_intent("When will it be ready?") == "timing"
    timing = handle_timing("")
    assert "pickup" in timing.lower()
