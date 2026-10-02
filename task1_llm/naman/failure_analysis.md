# Failure analysis: character-level GPT (run `main`)

All snippets are copied verbatim from `outputs/main_samples/` (best checkpoint, epoch 15, 830,976 parameters,
83-character vocabulary). Generation uses seed base 8893, 500 new characters per prompt.

## Case 1: Repetition loop (greedy decoding)

**File:** `outputs/main_samples/greedy.txt`, prompt 3 `'The big dog'`

> The big dog was so happy to have a big box. The box was so happy to have a big box. The box was so happy to have a big box. The box was so happy to have a big box. The box was so happy to have a big box.

The same sentence repeats until the 500-character limit. Prompt 1 shows the same loop starting midway:

> The boy was so happy to have a big box. The boy was so happy to have a big box. The box was so happy to have a big box. The box was so happy to have a big box.

**Failure type:** repetition / degenerate loop.

**Evidence:** the mean repeated 4-gram rate (word 4-grams, mean over the 5 prompts) is 0.599 for greedy decoding
(`repeated_4gram_rate` = 0.59879675 in `metrics_report.csv`). With sampling it is 0.0467 at temperature 0.7,
0.0168 at temperature 1.0 and 0 at temperature 1.2. Distinct-2 is 0.449 for greedy against 0.893 at temperature 0.7.

**Phrase counts (exact string matches, from `data_processed/train_ids.npy` and `val_ids.npy`, decoded with `vocab.json`;
train = 12,800,001 characters, validation = 1,280,001 characters):**

| phrase | train count | per 100K chars | validation count | per 100K chars | greedy samples (5 prompts) |
|---|---|---|---|---|---|
| "so happy to have" | 119 | 0.93 | 16 | 1.25 | 26 |
| "a big box" | 540 | 4.22 | 68 | 5.31 | 30 |
| "to have a big box" | 0 | 0.00 | 0 | 0.00 | 25 |
| "was so happy to have a big box" | 0 | 0.00 | 0 | 0.00 | 25 |

Both short phrases are rare in the training text (about 1 and 4 occurrences per 100K characters), and the full loop
sentence "...was so happy to have a big box" never occurs in the training or validation text. The model assembled the
loop sentence itself from two common fragments, and in the greedy samples "so happy to have" occurs 26 times and
"a big box" 30 times. Both appear in all 5 greedy prompts.

**Why it happens here:** greedy decoding always takes the single most likely next character. After "...to have a big
box." the most likely continuation is the sentence that has just been written, because the context window (128
characters) is almost entirely made of that sentence. Once the context is a repeated sentence, the model sees its
own output as strong evidence that the pattern should continue, so the loop feeds itself. A 0.83M-parameter model
has little capacity to track what the story has already said, and cannot plan a way out of the loop.

## Case 2: Broken words and grammar (high temperature)

**File:** `outputs/main_samples/temp_1.2.txt`, prompt 1 `'Once upon a time'`

> Once upon a time, there was a xect. They were sad busy. Whirl they were playing, they went home, and dary on, time they had a big kind and a nuts. One sunny day, Lily found a flower to the bubble. He wanted to lead a big, toast wish liked.
> "Come we go invite my raft towards?" her mom was an uolcause branchest.

**Failure type:** broken grammar and invented words ("xect", "dary", "uolcause", "branchest"), with
ungrammatical word order ("sad busy", "Come we go invite my raft towards?").

**Why it happens here:** a character-level model has to spell every word from individual characters, so a single
low-probability character early in a word can lead the word into a string that is not in the vocabulary of English.
Temperature 1.2 flattens the distribution, so such characters are sampled more often. The word-level structure is
also learned only implicitly, so grammar is weaker than spelling. At this temperature the output stays
punctuation- and capitalization-correct while the words themselves fall apart. The repeated 4-gram rate is 0
here (no repetition), and the errors are in spelling and word choice instead.

## Case 3: Loss of coherence / contradiction (greedy decoding)

**File:** `outputs/main_samples/greedy.txt`, prompt 5 `'Sara was sad because'`

> Sara was sad because she was so happy to have a big box. She was so happy to have a big box of fun.
> The box was so happy to have a big box. He wanted to play with the box and have fun.

**Failure type:** loss of coherence (a self-contradicting cause, an object that is "happy", and a pronoun that
changes from "She" to "He" with no new character introduced).

A related case with a sampled setting is `outputs/main_samples/temp_0.7.txt`, prompt 2:

> She looked at Lily and said, "I will share you around the tree." Lily was sad.

Here the model has the speaker "She" (the girl named Lily in the prompt) address "Lily", so the same character is
talking to herself.

**Phrase counts (same method as Case 1):** "was sad because" occurs 439 times in the training text (3.43 per 100K
characters) and 32 times in the validation text (2.50 per 100K). The string "sad because she was so happy" occurs 0 times in both.
"was so happy" occurs 2,115 times in the training text (16.52 per 100K) and 198 times in validation (15.47 per 100K),
and in the greedy samples 27 times.

**Why it happens here:** the prompt sets up a sad character, and in the training text "was sad because" is followed by a
reason for sadness (for example " he could not play with h...", " he lost his favorite toy"). The greedy output instead
continues with "she was so happy to have a big box", a combination that never occurs in the training text. With only
4 layers and a 128-character window (a couple of sentences), the model's prediction after "because" follows the frequent
"was so happy" pattern rather than the meaning of the prompt. This gives fluent sentences that do not agree with each other.
The model has no explicit representation of the characters in the story, so pronouns and names drift.
