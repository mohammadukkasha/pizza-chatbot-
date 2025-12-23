from dataclasses import dataclass, field
from typing import List

@dataclass
class OrderItem:
    category: str  # e.g., "pizza", "side", "drink"
    name: str
    size: str = ""
    crust: str = ""
    toppings: List[str] = field(default_factory=list)
    quantity: int = 1
    special: str = ""

@dataclass
class Order:
    items: List[OrderItem] = field(default_factory=list)
    notes: str = ""

    def add_item(self, item: OrderItem):
        self.items.append(item)

    def summary(self) -> str:
        lines = []
        for i, it in enumerate(self.items, 1):
            parts = [f"{i}. {it.quantity}x {it.name.title()} ({it.category})"]
            if it.size:
                parts.append(f"Size: {it.size}")
            if it.crust:
                parts.append(f"Crust: {it.crust}")
            if it.toppings:
                parts.append(f"Toppings: {', '.join(it.toppings)}")
            if it.special:
                parts.append(f"Special: {it.special}")
            lines.append(" | ".join(parts))
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)
