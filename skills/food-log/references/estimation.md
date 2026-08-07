# Estimation reference — portions, calories, quality, flags

Rough is fine. The goal is a consistent yardstick so week-over-week trends mean
something, not clinical accuracy. Use the same anchors every time.

## Portion anchors (visual)

- **Palm of a hand** ≈ 100–120 g cooked meat/fish (protein ~25–30 g).
- **Cupped hand / fist** ≈ 150–200 g cooked rice/pasta/potato, or one fruit.
- **Thumb** ≈ 1 tbsp fat (butter/oil ~100 kcal), or a portion of cheese.
- **Deck of cards** ≈ a 100 g meat portion.
- **Standard plate** full of one starch ≈ 300–400 g.
- Restaurant/takeaway portions and anything visibly fried: bias the estimate **up**.

## Calorie ready-reckoner (per common portion)

| Food | Portion | ≈ kcal |
|---|---|---|
| Cooked rice / pasta | 200 g | 260–280 |
| Bread | 1 slice (40 g) | 100 |
| Chicken breast, grilled | 150 g | 240 |
| Red meat / fatty cuts | 150 g | 350–450 |
| Salmon / oily fish | 150 g | 300 |
| Eggs | 2 large | 155 |
| Cheese | 30 g | 120 |
| Oil / butter (cooking) | 1 tbsp | 100–120 |
| Mixed salad, no dressing | bowl | 50–100 |
| Avocado | half | 160 |
| Nuts | small handful (30 g) | 180 |
| Potato, boiled | 200 g | 160 |
| Potato, fried / chips | 150 g | 400–500 |
| Sugary drink / juice | 330 ml | 140 |
| Beer | 500 ml | 210 |
| Wine | 150 ml | 125 |
| Pastry / dessert | 1 piece | 300–450 |
| Fast-food burger | 1 | 500–800 |

## Known products (exact label data — use these directly)

| Product | Portion | kcal | Protein | Carbs | Fat |
|---|---|---|---|---|---|
| Emmi Caffè Latte, Arabica, 281г | 1 cup | 200 | 10.2г | 15г | 3г |
| Emmi Caffè Latte, BIG, 400мл | 1 cup | 220 | 28г | 20г | 12г |
| Emmi Caffè Latte, Double Zero, 330мл | 1 cup | 200 | 26г | 12г | 5г |
| Wassertürmli (миндальное печенье) | 50г (1 порция) | 240 | 3.5г | 20.1г | 16.3г |
| Тоблерон, 1 конфетка | ~13г | 65 | 1.5г | 8г | 3.5г |

## Shelled nuts rule - stated weight includes shells

When the user gives a weight for in-shell nuts, log only the edible kernel share
and note it in `items`:

- **Pistachios:** kernel ≈ 50-55% (89г with shells ≈ 45-50г kernels)
- **Walnuts in shell:** kernel ≈ 40%
- **Pecans in shell:** kernel ≈ 35%

If the user says «без шелухи» / «очищенные», take the full stated weight. If
unsure, check the photo - unshelled pistachios are visually obvious.

## Front-label claims lie - trust the per-100g table

A front label's bold number is a per-serving claim with an ambiguous "serving",
not per-100g. Real case: Emmi Caffè Latte Protein displays "26g PROTEIN" but the
table says 3.45g/100g, so a 281g bottle holds ~10g. **Always compute from the
per-100g table × actual grams consumed**; when the user reads the table to you,
that is authoritative (confidence `high`).

Sum the components, then sanity-check against the plate as a whole. When torn
between two numbers, pick the middle and set `confidence: "low"`.

## Macro shortcuts

- Protein: 4 kcal/g · Carbs: 4 kcal/g · Fat: 9 kcal/g.
- Lean protein plate (meat/fish + veg): protein-heavy, low carb.
- Starch-forward plate (rice/pasta/bread/potato): carb-heavy.
- Fiber comes from vegetables, fruit, legumes, whole grains — near zero from
  refined/processed food.
- If you estimate kcal but are unsure of the split, approximate macros from the
  dominant components rather than leaving them all blank.

## Nutrient quality — the 1–10 scale

A single number for "how nourishing is this", independent of calories.

- **9–10** — whole foods, balanced: vegetables/fruit + lean protein + whole grains
  or legumes; minimal processing, little added sugar or refined oil.
- **7–8** — mostly whole foods with a minor caveat (a bit too much starch, some
  oil, one refined element).
- **5–6** — mixed: real food but noticeably refined, fried, or unbalanced (heavy
  carbs, low veg/protein).
- **3–4** — largely processed / fried / sugary with little nutritional upside.
- **1–2** — empty calories: sweets, sugary drinks, chips, pure junk.

Judge the **composition**, not the amount. A small dessert is still low quality;
a large healthy bowl is still high quality.

## `flags` — controlled vocabulary (use ONLY these)

Pick the few that apply. Consistency here is what makes the tag trends readable.

- `whole-food` — minimally processed, recognizable ingredients
- `ultra-processed` — packaged/industrial, long ingredient list
- `protein-rich` — a solid protein source present
- `veg` — meaningful vegetables on the plate
- `fruit` — fruit present
- `refined-carbs` — white bread/rice/pasta/sugar dominant
- `fried` — deep-fried or heavily pan-fried in oil
- `sugary` — added-sugar-forward (dessert, sweet drink)
- `high-sodium` — visibly salty / cured / processed meat / heavy sauce
- `alcohol` — contains alcohol
- `homemade` — clearly home-cooked
- `fast-food` — takeaway / fast-food origin

## `confidence`

- `high` — a single clearly-portioned item, or the user gave weights.
- `medium` — a normal plate you can read reasonably well.
- `low` — mixed dish, hidden ingredients, unclear portion, or ambiguous photo.

Be honest here — `low` is a perfectly good answer and keeps the whole system trustworthy.
