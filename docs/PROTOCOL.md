# Fixed endpoint and evaluation contract

Only professional Counter-Strike 2 Dust2 matches in the bundled fixed index
belong to this experiment. The main roles are 30 training, 10 development and
11 Rio test matches. Dates through 2026-04-06 are train, through 2026-04-10 are
development, and later included matches are test. The internal 22/8 assessment
uses only the main training cohort and holds out the last `ceil(n/5)` matches
within each event, ordered by `(date, str(match_id))`.

The observer is the CT team. Five CT slots are stable within a round; five
enemy slots are assigned by first public exposure. Enemy coordinates represent
the last public contact, not contemporaneous hidden replay truth. The source
parser builds memories, physical normalization and causal differences. Inputs
are `x` (frames × 10 × 32), `pub` (frames × 64) and frame game seconds. The
compact neural model transforms these into 272 dimensions; the risk and SITE
modules compute their own causal reductions. Feature definitions are in
`data/feature_schema.json`, with executable source in `sparc_cs2/features.py`,
`neural.py`, `risk.py`, `site.py` and the original parser.

Nine bits in the next event set are: 1 ingress A, 2 ingress B, 4 regroup A→B,
8 regroup B→A, 16 plant A, 32 plant B, 64 T death, 128 CT death, 256 round end.
Events at the same native tick are merged with bitwise OR. The target is the
first subsequent event; its time bin is right closed: `(j,j+1]` seconds,
`j=0,...,19`. `target_bin=20` is no event through 20 seconds, and `-1` denotes
unknown/unscorable. Mark values 1,...,511 index the dense event-set axis as
`mark-1`. The tail does not have a mark. `events`, `event_seconds`,
`event_time`, `ingress_first_time`, `target_bin` and `target_mask` are targets
or diagnostics. They are not fed into a forward model.

`scoring_eligibility` is unchanged from the original source. It checks finite,
strictly increasing observation game times, strictly ordered native ticks,
strictly increasing merged event game times, valid target shapes and at least
one eligible cutoff. Degenerate clocks exclude a whole round from every
method. `frame_indices` retains `cutoff_valid & (target_bin >= 0)`.

Joint NLL uses the probability at `(target_bin, target_mask-1)` for an event,
otherwise the tail. Time NLL sums over all 511 marks first. Ten-second A/B
ingress probabilities sum the first ten event bins over all masks with the
corresponding ingress bit, including compound events. Their Brier targets are
`(target_bin < 10) & (target_mask & bit != 0)`. Every metric averages cutoffs
within round, rounds within match, matches equally, then three independent
seed scores equally. The training weight is the same match/round/cutoff
balancing, multiplied by training row count to preserve optimizer loss scale.

The NN class law has eleven explicit masks and one catch-all component with
a training-only smoothed within-component law. The risk learner fits the
hazard using all positive exposures and a 25% sample of negative exposures,
with inverse sampling weights; its mark law uses five prospective time bands.
The SITE layer redistributes five groups using a fixed 50:50 conditional
mixture while preserving each time-bin mass and each group's internal law.
Public feasibility masks and fitted deadline min-coupling follow. A 2% source
component retains support. Nominal remaining time is `115-elapsed` before
plant and `40-plant_age` after plant; offsets are fit on training matches only.

One seed has one 55,577-parameter neural model. Shared boosted models comprise
105 binary hazard trees, 640 conditional-type trees, and 240 SITE trees: 985
total. Shuffled-control trees are separate control artifacts, excluded from
the deployed model count. Seed comparisons deploy one decoder at a time.

Saved destination diagnostics additionally condition on a pure entry/plant
family and its realized future bin, combine entry and plant masses, average
seed probabilities before log scoring, and deduplicate native events within
right-closed lead bands. These conditional diagnostics are not unconditional
forecast accuracy. Paired-effect intervals resample matches after averaging
seed differences, using the original 10,000 bootstrap draws.
