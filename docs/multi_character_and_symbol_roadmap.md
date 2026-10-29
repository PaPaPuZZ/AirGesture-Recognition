# Multi-Character, Word, Number, Symbol, and Diagram Roadmap

## Objective

Extend the IMU-based pen from isolated character recognition to:

1. multi-digit numbers;
2. words made from multiple letters;
3. a finite vocabulary of symbols;
4. eventually, simple diagrams or traced shapes.

These are different technical problems and should be developed in stages.

## Stage 1 — Isolated Character Recognition

Current work classifies one fixed recording window as one class:

```text
one gesture window -> one letter or digit
```

This stage must first become robust across:

- sessions;
- board poses;
- pen orientations;
- writing speeds and sizes;
- participants.

## Stage 2 — Character Segmentation and Sequence Assembly

The simplest practical route to words and larger numbers is to recognize one character at a time and append predictions to a sequence buffer:

```text
record character -> classify -> append -> repeat
```

Example:

```text
"2" + "0" + "2" + "6" -> "2026"
"P" + "E" + "N" -> "PEN"
```

The system needs a reliable way to identify character boundaries. Candidate boundary signals include:

- a button press;
- a dwell/stillness period;
- a deliberate separator gesture;
- nib contact or pressure threshold;
- explicit start/end commands in the application.

For a physical pen, the future differential-pressure nib sensor may provide a useful `pen_down` / `pen_up` signal, but pressure alone will not identify which character was written.

Recommended first interface:

```text
press/start command
-> draw one character
-> short stillness or release
-> classify
-> append to text buffer
```

Add user controls for:

- delete/backspace;
- space;
- clear sequence;
- end/submit.

## Stage 3 — Sequence-Level Decoding

After isolated predictions are reliable, add a lightweight decoder that uses context:

- allowable numeric formats;
- dictionary-based word correction;
- character confidence scores;
- language-model rescoring;
- beam search across several likely character predictions.

This decoder should correct uncertain sequences without hiding raw classifier confidence.

## Stage 4 — Continuous Unsegmented Writing

Recognizing a continuous word without explicit character boundaries is a harder sequence-recognition problem. It requires training examples labeled as whole sequences rather than isolated characters.

Possible later models include:

- temporal convolutional networks;
- bidirectional recurrent models;
- CTC-based sequence recognition;
- small transformers;
- encoder-decoder models.

This stage requires substantially more data and should not replace the isolated-character pipeline until the segmented approach is working reliably.

## Symbols

A finite symbol vocabulary is realistic with IMU data if each symbol is treated as another gesture class. Examples might include:

```text
+
-
=
?
!
arrow
circle
check mark
```

Each symbol needs:

- a consistent stroke convention;
- multiple sessions;
- multiple orientations;
- multiple participants;
- fresh holdout testing.

## Diagrams and Arbitrary Tracing

Classifying a known diagram or symbol is different from reconstructing an arbitrary drawing.

An IMU alone can classify motion patterns, but reconstructing accurate 2D or 3D paths from acceleration requires repeated integration and accumulates drift. Therefore, IMU-only tracking may not be sufficient for faithful free-form tracing.

Potential additional references include:

- an optical-flow sensor near the pen tip;
- a camera or external vision system;
- a tablet or touch-sensitive surface;
- magnetic or acoustic position tracking;
- surface markers or tracing templates.

The pressure sensor can provide nib-contact timing and force, but it does not provide absolute X/Y position.

## Recommended Development Order

1. Complete robust isolated digits and letters.
2. Validate board-pose and participant generalization.
3. Implement explicit one-character-at-a-time sequence assembly.
4. Add pressure-based or dwell-based character boundary detection.
5. Build multi-digit number and word demonstrations.
6. Add a small finite symbol vocabulary.
7. Evaluate an external position reference before attempting arbitrary diagrams or tracing.

## Success Criteria

A sequence demonstration should report both:

- character accuracy;
- whole-sequence accuracy.

For example, a four-character sequence is only completely correct when all four characters and their order are correct.
