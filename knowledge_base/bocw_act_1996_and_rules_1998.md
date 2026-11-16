# The Building and Other Construction Workers (Regulation of Employment and Conditions of Service) Act, 1996 & Central Rules, 1998

## Statutory Reference Corpus for RakshaSite Autonomous Compliance Agent

---

### Section 38 — Safety Committee and Safety Officers (BOCW Act, 1996)
1. **Section 38(1)(a)**: In every establishment wherein five hundred or more building workers are ordinarily employed, the employer shall constitute a Safety Committee consisting of such number of representatives of the employer and the building workers as may be prescribed by the State Government.
2. **Section 38(2)**: In every establishment referred to in sub-section (1), the employer shall also appoint a Safety Officer who shall possess such qualifications and perform such duties as may be prescribed, including enforcing mandatory Personal Protective Equipment (PPE) compliance such as industrial safety helmets (IS 2925:1984) and high-visibility reflective safety vests on active construction zones.
- **Violation Mapping**: `no_helmet`, `NO_HELMET`, `no_vest`, `NO_VEST`, `missing_ppe`
- **Statutory Penalty**: Contravention punishable under Section 47 with imprisonment up to 3 months, or fine up to Rs. 2,000, or both.

---

### Section 39 — Notice of Certain Accidents & Elevated Work Hazards (BOCW Act, 1996)
1. **Section 39(1)**: Where in any establishment an accident occurs which causes death or which causes any bodily injury by reason of which the person injured is prevented from working for a period of forty-eight hours or more immediately following the accident, the employer shall give notice thereof to the prescribed authority.
2. **Fall & Elevated Work Mandate**: Read with BOCW Central Rules 1998 (Rule 41 & Rules 175–185) and IS 3696 / IS 3521, all workers operating on scaffolding, unguarded edges, or elevations exceeding 2 meters must be equipped with full-body fall-arrest harnesses, lifelines, and safety nets.
- **Violation Mapping**: `no_harness`, `NO_HARNESS`, `fall_hazard`, `scaffolding_elevated`
- **Statutory Penalty**: Contravention punishable under Section 47 with imprisonment up to 3 months and/or fine up to Rs. 2,000 (up to 6 months and Rs. 4,000 for repeat offences).

---

### Section 40 — Power of Appropriate Government to Make Rules for Safety and Health (BOCW Act, 1996)
1. **Section 40(1)**: The appropriate Government may, by notification, make rules regarding the measures to be taken for the safety and health of building workers in the course of their employment, and the equipment and appliances necessary to be provided to them for ensuring their safety, health and protection.
2. **Section 40(2)(k)**: Precautions to be taken with respect to scaffolds, ladders, stages, and temporary structures (IS 3696).
3. **Section 40(2)(o)**: Safety precautions for preventing falls of persons from height, including safety harnesses and safety nets (IS 3521).
4. **Section 40(2)(r)**: Safety precautions and provision of personal protective equipment including head protection helmets (IS 2925), high-visibility safety vests, protective footwear, eye goggles, and industrial gloves.
- **Violation Mapping**: `no_helmet`, `no_vest`, `no_harness`, `no_gloves`, `no_boots`
- **Statutory Penalty**: Section 47 — Fine up to Rs. 2,000 and/or imprisonment up to 3 months; Rs. 100/day for continuing contravention.

---

### Rule 46 — Head Protection / Industrial Safety Helmets (BOCW Central Rules, 1998)
Every building worker required to pass through or work within areas where there is a hazard of being struck by falling or flying objects or electrical shock shall be provided with and shall wear an industrial safety helmet conforming to Bureau of Indian Standards specification **IS 2925:1984**.
- **Violation Mapping**: `no_helmet`, `NO_HELMET`
- **Statutory Citation**: `Section 40(2)(r) & BOCW Central Rule 46 (IS 2925:1984)`
- **Statutory Penalty**: Section 47 — Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.

---

### Rule 45 & Rule 54 — High-Visibility Safety Vests & Protective Apparel (BOCW Central Rules, 1998)
Every building worker working near moving vehicles, cranes, excavators, haul roads, or low-visibility construction zones shall be provided with and shall wear high-visibility reflective safety vests (fluorescent orange/yellow with retro-reflective tape) to prevent struck-by-machinery accidents.
- **Violation Mapping**: `no_vest`, `NO_VEST`
- **Statutory Citation**: `Section 38(1)(a), Section 40(2)(r) & BOCW Central Rule 45`
- **Statutory Penalty**: Section 47 — Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.

---

### Rule 49 — Hand and Foot Protection: Gloves & Safety Boots (BOCW Central Rules, 1998)
1. **Foot Protection**: Every building worker engaged in construction, concreting, demolition, or material handling shall be provided with protective safety footwear (steel-toe boots conforming to IS 15298) to prevent puncture and crushing injuries.
2. **Hand Protection**: Suitable protective gloves (leather, insulated, or chemical-resistant) shall be provided for sharp materials, welding, or electrical work.
- **Violation Mapping**: `no_gloves`, `NO_GLOVES`, `no_boots`, `NO_BOOTS`
- **Statutory Citation**: `Section 40(2)(r) & BOCW Central Rule 49`
- **Statutory Penalty**: Section 47 — Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.

---

### Rule 175 to 185 & IS 3696 — Scaffolding, Ladders & Fall Arrest Systems (BOCW Central Rules, 1998)
Every scaffold platform above 2 meters height shall be provided with rigid guardrails, toe-boards, and safe access ladders conforming to **IS 3696 (Parts 1 & 2)**. Workers on elevated scaffolding or unguarded edges must wear a full-body safety harness conforming to **IS 3521** anchored to an independent lifeline.
- **Violation Mapping**: `no_harness`, `NO_HARNESS`, `scaffolding_elevated`
- **Statutory Citation**: `Section 40(2)(k), Section 40(2)(o) & BOCW Central Rules 175–185 (IS 3696 / IS 3521)`
- **Statutory Penalty**: Section 47 — Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.

---

### Section 44 — Responsibility of Employers (BOCW Act, 1996)
An employer shall be responsible for providing constant and adequate supervision of any building or other construction work in his establishment as to ensure compliance with the provisions of this Act relating to safety and for taking all practical steps necessary to prevent accidents.
- **Violation Mapping**: `unsafe_environment`, `failure_to_supervise`, `repeated_violation`
- **Statutory Penalty**: Section 47 & Section 50 — Fine up to Rs. 2,000 and/or imprisonment up to 3 months, plus potential suspension of establishment registration.

---

### Section 47 — Penalty for Contravention of Safety Provisions (BOCW Act, 1996)
1. **Section 47(1)**: Whoever contravenes the provisions of any rules made under Section 40 shall be punishable with imprisonment for a term which may extend to three months, or with fine which may extend to two thousand rupees, or with both, and in the case of a continuing contravention, with an additional fine which may extend to one hundred rupees for every day during which such contravention continues after conviction for the first such contravention.
2. **Section 47(2)**: If any person who has been convicted of any offence punishable under sub-section (1) is again guilty of an offence involving a contravention of the same provision, he shall be punishable on a subsequent conviction with imprisonment for a term which may extend to six months and with fine which shall not be less than five hundred rupees but which may extend to two thousand rupees.
