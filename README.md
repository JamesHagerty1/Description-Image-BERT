self attention on sequences of form <card name tokens><card image byte tokens>
trained with cloze task, e.g. '9 of spades' in training batch appears as '[MASK] of spades' and '9 of [MASK]'
see attn_cards folder, visualizes three card name tokens' attention to card image bytes

test