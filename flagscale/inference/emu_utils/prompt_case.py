T2I_CASE = [
    {
        "prompt": """A lively comic-style illustration depicting two humorous cartoon dogs interacting near a freshly dug backyard hole surrounded by scattered dirt, garden tools, blooming flowers, and a wooden fence background. At the upper-left side, Dog One stands nervously near the messy hole, ears down and eyes wide open with an expression of concern. Its speech bubble is an oval shape, outlined neatly with smooth, slightly rounded corners, positioned clearly above Dog One's head. Inside, clearly readable playful handwritten-style text emphasizes the dog's worried tone, saying, "You sure the humans won't notice this giant hole here?". Toward the lower-right side, Dog Two sits calmly and confidently with a cheerful, carefree expression, wagging its tail gently. Its speech bubble is rectangular with softly rounded edges, placed slightly overlapping with Dog One's speech bubble to guide the reader naturally downward diagonally across the frame. Dog Two's friendly, humorous response appears in a whimsical italicized comic font, clearly stating, "Relax! We'll just blame it on the neighbor's cat again!". Each speech bubble creates the playful and engaging backyard scene.""",
        "reference_image": [],
    }
]


X2I_CASE = [
    {
        "prompt": "As shown in the second figure: The ripe strawberry rests on a green leaf in the garden. Replace the chocolate truffle in first image with ripe strawberry from 2nd image",
        "reference_image": ["./assets/ref_0.png", "./assets/ref_1.png"],
    }
]


HOWTO_CASE = [{"prompt": "How to cook Shrimp, Celery, and Pork Dumplings.", "reference_image": []}]


STORY_CASE = [
    {
        "prompt": "Imagine a heartwarming tale about a little hedgehog who overcomes his fear of the dark with the help of glowing fireflies.",
        "reference_image": [],
    }
]


EMU_TASKS = {"t2i": T2I_CASE, "x2i": X2I_CASE, "howto": HOWTO_CASE, "story": STORY_CASE}
