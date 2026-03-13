## 5.1 Audio CD Subcode.

Jasper gedaan

### Q1: Subcode Data Rates

- **Source:** _BS EN 60908:1999_, Pages 17–18.
- **Block Frequency:** One subcoding block consists of **98 symbols**. The repetition frequency is **75 Hz**.
- **Bits per Frame:** After demodulation, **8 bits** per frame are used.

**Calculations:**

- **Bits per Block:** $98 \times 8 = 784$ bits
- **Total Bitrate:** $75 \text{ Hz} \times 784 \text{ bits} = \mathbf{58,800 \text{ bits/s}}$

### Q2: Channel Distribution

- **Source:** _Ken C. Pohlmann, The Compact Disc Handbook (1989)_, Page 93.
- **Reference:** See image `5.1 Q2.png`.

### Q3: Table of Contents (TOC) & Error Protection

How does the CD player determine track counts, start positions, and total time?

- **Source:** _Ken C. Pohlmann, The Compact Disc Handbook (1989)_, Page 94-95.

> **Mechanism:** The player reads the **Table of Contents (TOC)** stored in the **Q-channel** of the **Lead-in area** during disc initialization.

| Data Type              | `POINT` Field | Description                                                         |
| :--------------------- | :------------ | :------------------------------------------------------------------ |
| **Number of Tracks**   | `A1` (Hex)    | The `PMIN` field records the track number of the final track.       |
| **Start Positions**    | `01` to `99`  | `PMIN`, `PSEC`, and `PFRAME` indicate the start time of that track. |
| **Total Playing Time** | `A2` (Hex)    | Indicates the absolute start time of the **Lead-out** track.        |

#### Error Protection Strategy

Because the TOC is in the **Lead-in area**, it is not protected by the standard **CIRC** (Cross-Interleaved Reed-Solomon Code) used for audio data. Instead, it uses:

- **CRC (Cyclic Redundancy Check):** Present in every subcode block to detect errors.
- **Redundancy:** Each track starting time is repeated in **three successive blocks**.
  The entire Table of Contents is repeated continuously through the Lead-in area to ensure the player eventually captures the data even if some blocks are corrupted.

## 5.2 Reed-Solomon code

1 Teun begint
2

## 5.3 Audio encoding in CDs

1 Milan mee bezig

2 Milan mee bezig

3
4
5
6
