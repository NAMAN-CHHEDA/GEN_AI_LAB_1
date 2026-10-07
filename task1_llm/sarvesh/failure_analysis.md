# Failure analysis — Sarvesh Task 1 (run `sarvesh_main`)

Snippets are verbatim from `outputs/sarvesh_main_samples/` (best checkpoint, epoch 12,
2.27M parameters, 85-character vocabulary).

## Case 1: Repetition loop (greedy decoding)

**Setting / file:** `outputs/sarvesh_main_samples/greedy.txt`  
**Prompt:** `Once upon a time`

**Snippet:**

> Lily said, "Lily, you are sorry. You are very kind and shared the treat. You are a good friend. You are very kind and shared the treat. You are a good friend. You are very kind and shared the treat.

A second clear loop appears under prompt `The big dog`:

> The big dog was sad. The dog was sad. The dog was sad. The dog wanted to help the dog. ... The dog said, "You are a good friend. You are very kind and happy." The dog was happy and said, "You are a good friend. You are very kind and happy."

**Failure type:** repetition / degenerate loop

**Observation:** Greedy decoding always picks the single most likely next character. Once a short high-probability phrase enters the 160-character context window, that phrase dominates the context and the model keeps regenerating it. Quantitatively, greedy has a repeated 4-gram rate of **0.277** vs **0.000–0.052** under sampling settings. Temperature / top-p largely remove the loop but trade fluency for other errors below.

## Case 2: Broken grammar / invented words (temperature sampling)

**Setting / file:** `outputs/sarvesh_main_samples/temp_1.0.txt`  
**Prompt:** `Once upon a time`

**Snippet:**

> Once upon a time, there was a popur tree. In a farm with a sink in his yard. The big tree liked to explore his lunch. The tree was a very alert. One day, the tree and the tree went for a walk. It was wide and closed.
> The tree started to grow bigger and lowers. It flew down to build a big soft box of colorful sword.

**Failure type:** broken grammar / invented words (spelling)

**Observation:** Character-level models must spell every word one character at a time. At temperature 1.0 a single unlikely early character ("popur" instead of "popular"/"poplar") sends the word into a non-English string. Grammar also drifts ("was a very alert", "bigger and lowers"). Distinct-n is high here (diversity looks good) while local linguistic quality collapses — the opposite failure mode of greedy.

## Case 3: Loss of coherence / contradiction (greedy)

**Setting / file:** `outputs/sarvesh_main_samples/greedy.txt`  
**Prompt:** `Sara was sad because`

**Snippet:**

> Sara was sad because she was so happy to have her mom and dad. She was very happy and said, "Thank you, Tom! You are very kind and helpful."
> Tom and Sara are sad. They like to play in the park. They have a lot of fun. They are happy.

Related self-reference under prompt `Once upon a time`:

> Lily said, "Lily, you are sorry.

**Failure type:** loss of coherence / contradiction (also weak character tracking)

**Observation:** The prompt sets a sad cause, but the continuation jumps to "so happy" — a fluent but contradictory clause. Pronouns and names also drift (Sara thanks an unexplained Tom; Lily addresses herself as "Lily"). With only 5 layers and a 160-character window, the model follows local n-gram patterns more than story-level emotion or speaker identity.
