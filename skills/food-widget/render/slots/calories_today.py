"""calories_today slot: kcal eaten today vs daily budget, meals, macros.

Payload schema (v1):
{
  "schema": 1,
  "date": "2026-07-07",
  "updated": "2026-07-07T14:58:00+02:00",   # added by widget_push if absent
  "budget_kcal": 2400,
  "eaten_kcal": 1830,
  "macros": {"protein_g": 92, "fat_g": 61, "carb_g": 168},
  "meals": [{"time": "08:40", "kcal": 520, "name": "Завтрак",
             "item": "овсянка, кофе"}, ...]
}
Layout is computed as a vertical flow - никогда не хардкодить координаты.
"""
from datetime import datetime

from PIL import Image

from ..card import fit_font, kcal_fmt, new_card, pill_bar, truncate
from ..theme import THEMES, font

TITLE = "Калории сегодня"
TTL_MINUTES = 240


def _status(eaten: float, budget: float, T: dict):
    rem = budget - eaten
    if eaten <= 0.85 * budget:
        color = T["good"]
    elif eaten <= budget:
        color = T["warn"]
    else:
        color = T["bad"]
    text = (f"осталось {kcal_fmt(rem)} ккал" if rem >= 0
            else f"перебор {kcal_fmt(-rem)} ккал")
    return text, color


def _updated_hhmm(p: dict) -> str:
    try:
        return datetime.fromisoformat(p["updated"]).strftime("%H:%M")
    except Exception:
        return ""


def _meal_ticks(p: dict):
    budget = max(p.get("budget_kcal") or 1, 1)
    ticks, cum = [], 0
    for meal in p.get("meals", []):
        cum += meal.get("kcal") or 0
        ticks.append(cum / budget)
    return ticks[:-1] if ticks else []


def _core(d, p, T, x, y, w, h, compact: bool):
    """Header, big number, status, bar, macros row - vertical flow in box."""
    eaten, budget = p["eaten_kcal"], p["budget_kcal"]
    s = {  # font sizes tuned per box class
        "head": 40 if compact else 46,
        "num": 170 if compact else 205,
        "unit": 48 if compact else 60,
        "status": 46 if compact else 58,
        "bar": 30 if compact else 38,
        "bottom": 34 if compact else 42,
    }
    # vertical flow with proportional gaps; shrink everything if it overflows
    need = (s["head"] + s["num"] + s["status"] + s["bar"] + s["bottom"])
    gaps = [int(s["num"] * g) for g in (0.16, 0.14, 0.15, 0.14)]
    total = need + sum(gaps)
    if total > h:
        k = h / total
        s = {k2: int(v * k) for k2, v in s.items()}
        gaps = [int(g * k) for g in gaps]

    cy = y
    # header (per-person slot title arrives via payload["_title"])
    head_f = font(s["head"], "medium")
    d.text((x, cy), truncate(d, p.get("_title") or TITLE, head_f, int(w * 0.66)),
           font=head_f, fill=T["sub"], anchor="la")
    hhmm = _updated_hhmm(p)
    if hhmm:
        d.text((x + w, cy), hhmm, font=font(s["head"], "regular"),
               fill=T["faint"], anchor="ra")
    cy += s["head"] + gaps[0]

    # big number + unit on the shared baseline
    num = kcal_fmt(eaten)
    f_num, num_size = fit_font(d, num + "  ккал", s["num"], "bold", w)
    f_unit = font(max(int(num_size * 0.30), 30), "regular")
    base_y = cy + num_size
    d.text((x, base_y), num, font=f_num, fill=T["text"], anchor="ls")
    num_w = d.textlength(num, font=f_num)
    d.text((x + num_w + int(num_size * 0.12), base_y), "ккал",
           font=f_unit, fill=T["faint"], anchor="ls")
    cy = base_y + gaps[1]

    # status line
    text, color = _status(eaten, budget, T)
    d.text((x, cy), text, font=font(s["status"], "semibold"),
           fill=color, anchor="la")
    cy += s["status"] + gaps[2]

    # bar + bottom row are pinned to the bottom of the box
    bottom_y = y + h - s["bottom"]
    bar_y = max(cy, bottom_y - gaps[3] - s["bar"])
    pill_bar(d, x, bar_y, w, s["bar"], eaten / max(budget, 1), color, T,
             ticks=_meal_ticks(p))
    cy = bottom_y

    # macros left, budget right
    m = p.get("macros") or {}
    if any(m.get(k) for k in ("protein_g", "fat_g", "carb_g")):
        macro = (f"Б {round(m.get('protein_g') or 0)}"
                 f"   Ж {round(m.get('fat_g') or 0)}"
                 f"   У {round(m.get('carb_g') or 0)}")
        d.text((x, cy), macro, font=font(s["bottom"], "regular"),
               fill=T["sub"], anchor="la")
        d.text((x + w, cy), f"из {kcal_fmt(budget)}",
               font=font(s["bottom"], "regular"), fill=T["faint"], anchor="ra")
    else:
        d.text((x, cy), f"бюджет {kcal_fmt(budget)} ккал",
               font=font(s["bottom"], "regular"), fill=T["sub"], anchor="la")
    return cy + s["bottom"]


def _meal_list(d, p, T, x, y, w, h):
    meals = p.get("meals", [])
    if not meals:
        d.text((x, y + 8), "Сегодня ещё ничего не записано",
               font=font(42, "regular"), fill=T["faint"], anchor="la")
        return
    s_txt = 44
    row_h = int(s_txt * 1.85)
    max_rows = max(1, h // row_h)
    show = meals[-max_rows:] if len(meals) > max_rows else meals
    hidden = len(meals) - len(show)
    time_w = int(d.textlength("00:00", font=font(s_txt - 6, "regular"))) + 36
    for i, meal in enumerate(show):
        ry = y + i * row_h
        kcal_s = kcal_fmt(meal.get("kcal") or 0)
        kcal_w = d.textlength(kcal_s, font=font(s_txt, "semibold"))
        d.text((x, ry), meal.get("time", ""), font=font(s_txt - 6, "regular"),
               fill=T["faint"], anchor="la")
        name = str(meal.get("name") or "Приём пищи")
        item = str(meal.get("item") or "").strip()
        f_name = font(s_txt, "medium")
        name_max = int(w - time_w - kcal_w - 40)
        if item:
            name_part = truncate(d, name, f_name, int(name_max * 0.55))
            d.text((x + time_w, ry), name_part, font=f_name,
                   fill=T["text"], anchor="la")
            used = d.textlength(name_part, font=f_name)
            rest = name_max - used - 18
            if rest > 60:
                d.text((x + time_w + used + 18, ry),
                       truncate(d, "· " + item, font(s_txt - 6, "regular"), int(rest)),
                       font=font(s_txt - 6, "regular"), fill=T["faint"], anchor="la")
        else:
            d.text((x + time_w, ry), truncate(d, name, f_name, name_max),
                   font=f_name, fill=T["text"], anchor="la")
        d.text((x + w, ry), kcal_s, font=font(s_txt, "semibold"),
               fill=T["text"], anchor="ra")
        if i < len(show) - 1:
            ly = ry + row_h - int(row_h * 0.22)
            d.line([x + time_w, ly, x + w, ly], fill=T["hairline"], width=2)
    if hidden > 0:
        d.text((x + time_w, y + len(show) * row_h - 6),
               f"+ ещё {hidden}", font=font(s_txt - 8, "regular"),
               fill=T["faint"], anchor="la")


def render(p: dict, w: int, h: int, theme: str) -> Image.Image:
    T = THEMES[theme]
    img, d = new_card(w, h, T)
    pad = int(min(w, h) * 0.10)
    inner_w, inner_h = w - 2 * pad, h - 2 * pad

    if w == h and w >= 1000:  # 4x4: core on top + meal list below
        core_h = int(inner_h * 0.47)
        used = _core(d, p, T, pad, pad, inner_w, core_h, compact=False)
        gap = int(inner_h * 0.055)
        d.line([pad, used + gap // 2, pad + inner_w, used + gap // 2],
               fill=T["hairline"], width=2)
        _meal_list(d, p, T, pad, used + gap, inner_w, (h - pad) - (used + gap))
    else:
        _core(d, p, T, pad, pad, inner_w, inner_h, compact=(w < 900))
    return img
