# Completeness of the selection algebra

A finished TorchLens capture is a finite directed acyclic graph of executed operations, and a
selection names a set of tensor elements inside it. This page states and proves what the
selection algebra can and cannot express. On one capture, every set of elements is the value of
some selection (absolute completeness). Every query in a declared language of attribute
predicates, set operations and bounded or unbounded reachability along dataflow is expressed by
one selection that does not depend on the capture it is later resolved against, with two
spelled-out caveats (relative completeness). The graph closure producers `tl.neighborhood` and
`tl.between` are necessary: no combination of attribute selectors and boolean operators, not even
full first-order logic, expresses reachability (necessity). The page defines the objects, states
the theorems, proves them, lists the code facts the proofs depend on, and works three examples
on a six-op model; the examples run as written in
`tests/test_selection_completeness_examples.py`.

## 0. The claim in one paragraph

A *selection* names a set of tensor elements inside the capture graph, together with the set of
operation outputs ("sites") it touches. The selection algebra has three parts: *generators*
(selectors that pick sites by their attributes, producers that pick elements by coordinates or
by values), *boolean combinators* (`|`, `&`, `-`, `~`), and *graph closure producers*
(`tl.neighborhood`, `tl.between`). We prove three things. First, on any one capture every set of
elements whatsoever is the value of some selection (absolute completeness). Second, every query
written in a declared query language, with attribute predicates, set operations (complement and
difference included, also inside a reachability operator) and bounded or unbounded reachability
along dataflow, is expressible by one selection that is independent of the capture it is later
resolved against, with two spelled-out caveats (relative completeness, the substantive claim).
Third, the closure producers are necessary: no combination of attribute selectors and booleans,
not even with full first-order quantification, expresses reachability (necessity). Completeness
is a claim about *where*; it says nothing about how much any site influences another.

## 1. Definitions

### 1.1 The trace graph

**Definition 1 (trace graph).** A finished capture determines a finite structure

```text
T = (V, E, attr, shape)
```

- `V` is the finite set of executed operation records with a string label: the computed ops
  in `trace.ops` plus the model's input and output records, the universe that
  `trace[tl.all_sites()].labels` lists in execution order. Labels are *pass-qualified*: a module executed twice contributes two records,
  `tanh_1_2:1` and `tanh_1_2:2`. So `V` is the unrolled execution, and `(V, E)` is acyclic. The
  model's inputs and outputs appear as records too (`input_1:1`, `output_1:1`).
- `E ⊆ V × V` is the recorded dataflow relation: `(u, v) ∈ E` when `u` is listed among `v`'s
  parents (`op.parents`). Write `children(u) = {v : (u, v) ∈ E}` and
  `parents(v) = {u : (u, v) ∈ E}`.
- `attr(v)` is the record's graph-free data: function name, module membership, label, pass
  index, execution index, input or output role, dtype, and the like. Some record fields are
  *graph-derived* (`parents`, `children`, `has_output_descendant`, `min_distance_from_input` and
  others). They are not part of `attr`; Section 4 explains why this matters.
- `shape(v)` is the output index space of `v`, a tuple of non-negative integers, defined when the
  output is one tensor. Multi-output functions such as `torch.max(x, dim)` are recorded as one
  record per output slot, each with its own shape.

**Definition 2 (sites and elements).** A *site* is a record `v ∈ V` with a defined `shape(v)`.
Write `V_T` for the set of sites. The *index space* of a site is

```text
Idx(v) = [0, d_1) × ... × [0, d_n)   where shape(v) = (d_1, ..., d_n).
```

The *element universe* is the disjoint union `U_T = ⊔_{v ∈ V_T} Idx(v)`; an element is a pair
`(v, i)` with `i ∈ Idx(v)`. Parameters and dataflow edges form two further sorts of site (kinds
`PARAM` and `EDGE`); they have their own universes and never mix with activation sites (kind
`ACT`) inside one selection. This page treats the `ACT` sort; Section 6 says what carries over.

### 1.2 Selections: the two-level denotation

**Definition 3 (resolved selection).** A resolved selection of kind `ACT` over `T` is a pair

```text
S = (F, m)
```

where `F ⊆ V_T` is a finite set of *touched* sites (ordered by execution in the implementation;
the order is not part of the mathematics) and `m` assigns to each `s ∈ F` a subset
`m(s) ⊆ Idx(s)`, the *mask*. A site may be touched with an empty mask: `s ∈ F` with `m(s) = ∅`
is legal and is different from `s ∉ F`.

Three readings of a selection are used below:

```text
Fam(S)   = F                              (the touched family)
Sites(S) = { s ∈ F : m(s) ≠ ∅ }           (the sites that carry at least one element)
El(S)    = { (s, i) : s ∈ F, i ∈ m(s) }   (the element set)
```

A selection is *whole-site* when every mask is either all of `Idx(s)` or empty, and
*family-clean* when every mask is all of `Idx(s)`; for family-clean selections
`Fam(S) = Sites(S)`.

**Definition 4 (query and resolution).** A *selector sentence* (class `tl.Selector`) is a finite
syntax tree built by the grammar of Section 1.3. It contains no reference to any capture. The
*resolution door* `trace[q]` maps a sentence `q` and a capture `T` to a resolved selection,
written `[[q]]_T`, or refuses with a typed error. Refusal is part of the semantics: `[[·]]_T` is a
partial function, and a sentence never silently narrows to "what could be resolved".

### 1.3 The grammar of the kernel K

Every atom below denotes a resolved selection on a capture; combinators and closure producers
build larger sentences.

**Atoms (generators).**

- *Attribute selectors*, written `lift(σ)` for a structural selector `σ`: `tl.label(name)`,
  `tl.func(name)`, `tl.func_transform(kind)`, `tl.module(address)`, `tl.in_module(address)`,
  `tl.label_contains(text)`, `tl.label_matches(regex)`, `tl.all_sites()`, `tl.site(...)`,
  `tl.model_input(...)`, `tl.model_output(...)`, the backward-graph family (`tl.grad_fn`,
  `tl.grad_input`, `tl.grad_output`, `tl.in_backward_pass`), `tl.at_step`, `tl.facet`,
  `tl.head`, `tl.without_op`. Structural selectors form their own boolean algebra (`σ & σ'`,
  `σ | σ'`, `~σ`, `σ - σ'`) *before* lifting; `~σ` there is predicate negation. Lifting is
  `σ.__selection__()`; `trace[σ]` lifts and resolves in one step.
- *Handle lifts*: `op.__selection__()` (one record, one pass), `layer.__selection__()` (one
  layer, all passes), `slice.__selection__()` for a `TraceSlice`.
- *Coordinate producers*: `tl.units(site, indices)` (explicit coordinates or a boolean mask at
  one site).
- *Value producers*, reading the saved activations of the resolution capture: `tl.top_k`,
  `tl.top_fraction`, `tl.threshold`, `tl.sign`; *statistical producers* reading an explicit
  sample set: `tl.dead`, `tl.saturated`, `tl.low_variance`; *comparative producers* reading one
  reference capture: `tl.changed`, `tl.top_changed`, `tl.stable_across_passes`,
  `tl.pass_variance`.
- *Geometry producers*: `tl.subspace(within, basis, origin=...)` (the support set of a
  direction), receptive-field and gradient-support boxes lifted by `__selection__`.
- *Control*: `tl.random_like(like=, within=, seed=)`.
- *Opaque predicate*: `tl.where(f)`, where `f` is any Python function of the full record. It is
  marked non-portable by the code and is **excluded from the kernel `K`** of the theorems below;
  Section 4 explains why.

**Combinators.** For sentences `A`, `B` of the same kind: `A | B`, `A & B`, `A - B`, `~A`. There
is deliberately no `^`; `(A - B) | (B - A)` spells it.

**Closure producers.** `tl.neighborhood(X, hops=k, direction=d)` with `k` a non-negative integer
and `d ∈ {'upstream', 'downstream', 'both'}`, and `tl.between(A, B)` where `A` and `B` are
sentences or lists of sentences.

`K` is the set of sentences built from these atoms, combinators and closure producers. `K0 ⊂ K`
is the sub-kernel with no closure producers.

### 1.4 The semantics function

Fix a capture `T`. Each rule below is what the code computes; the code pointers are in Section 4.

**Atoms.**

- `[[lift(σ)]]_T = (M, s ↦ Idx(s))` where `M = {v ∈ V : σ holds of v}`; every matched site gets a
  full mask. If some matched record has no `shape`, resolution refuses (`non_tensor_site` or
  `no_index_space`) rather than dropping it.
- `[[op.__selection__()]]_T = ({v}, v ↦ Idx(v))` for the record `v`.
- `[[tl.units(s, I)]]_T = ({s}, s ↦ I)` for `I ⊆ Idx(s)`.
- Value, statistical, comparative and geometry producers denote `(P, s ↦ m(s))` where `P` is the
  population named by `within=` (default: every site with a saved tensor value) and `m(s)` is the
  set of elements satisfying the producer's criterion. Their exact criteria are not needed below;
  what matters is that each is a function of `attr`, of coordinates, of saved values, or of the
  explicit evidence captures, and of nothing else.

**Combinators.** With `[[A]]_T = (F_A, m_A)` and `[[B]]_T = (F_B, m_B)`:

```text
[[A | B]]_T :  F = F_A ∪ F_B,   m(s) = m_A(s) ∪ m_B(s)   (a missing side counts as ∅)
[[A & B]]_T :  F = F_A ∩ F_B,   m(s) = m_A(s) ∩ m_B(s)
[[A - B]]_T :  F = F_A,         m(s) = m_A(s) \ m_B(s)   (subtraction never un-touches)
[[~A]]_T    :  F = F_A,         m(s) = Idx(s) \ m_A(s)   (complement per touched site)
```

So for a fixed family `F` the masks form the product boolean algebra `∏_{s ∈ F} 2^{Idx(s)}`,
while the family itself only ever grows by `|`, shrinks by `&`, and is left alone by `-` and `~`.
In particular `~` is **not** complement in the universe `V_T`: there is no sentence `~A` whose
touched family is `V_T \ Fam(A)`. Complement in the universe is spelled `tl.all_sites() - A`,
whose family is all of `V_T` and whose sites are `V_T \ Sites(A)` when `A` is whole-site.

**Closure producers.** Write `Reach_d^{≤k}(X)` for the set of records reachable from the set
`X ⊆ V` by a directed path of at most `k` edges in direction `d` (`downstream` follows
`children`, `upstream` follows `parents`, `both` follows either at every step), with
`X ⊆ Reach_d^{≤0}(X) = X`. Write `Desc(X) = Reach_downstream^{≤|V|}(X)` and
`Anc(X) = Reach_upstream^{≤|V|}(X)`.

```text
[[tl.neighborhood(X, hops=k, direction=d)]]_T = (Reach_d^{≤k}(Sites([[X]]_T)),  s ↦ Idx(s))
[[tl.between(A, B)]]_T = (Desc(Sites([[A]]_T)) ∩ Anc(Sites([[B]]_T)),  s ↦ Idx(s))
```

Two facts about these rules carry the proof. First, the seed of a closure producer is the set of
sites of its operand that carry at least one element: a site with a partial mask seeds its whole
node, and a touched site whose mask is empty seeds nothing. Second, the result is always
family-clean. If a reached record has no `shape`, resolution refuses rather than dropping it. The
graph view `trace.between(A, B)` and `trace.subgraph(X)` use the same member sets.

### 1.5 The query language Q

"Complete" is always relative to a declared class of queries. The class is the natural
"where-clause" language over one capture: sets of sites defined by attributes, set operations
and reachability.

**Definition 5 (site queries).** Fix a family `P` of unary *attribute predicates*: any predicate
on records computable from `attr(v)` alone. Site queries are generated by

```text
q ::= α                       α ∈ P
    | q ∪ q | q ∩ q | q \ q | ¬q
    | Reach_d^{≤k}(q)         d ∈ {fwd, bwd, both}, k ≥ 0
    | Reach_d^{*}(q)          unbounded
```

with semantics `[[α]]_T = {v ∈ V_T : α(attr(v))}`, set operations on `V_T` (so `¬q` is
`V_T \ [[q]]_T`), `[[Reach_d^{≤k}(q)]]_T = Reach_d^{≤k}([[q]]_T) ∩ V_T` and
`[[Reach_d^{*}(q)]]_T = ⋃_k [[Reach_d^{≤k}(q)]]_T`. Up to notation this is the monadic fragment
of first-order logic with transitive closure over the vocabulary `(E, P)`.

**Definition 6 (element queries).** Relative to a site query `q`, element queries are boolean
combinations (`∪ ∩ \` and complement within `⊔_{s ∈ [[q]]} Idx(s)`) of *element predicates*:
coordinate predicates on `Idx(s)` and value predicates on saved activations (of the resolution
capture, a reference capture, or a sample set). A full query is a pair `(q, e)`.

## 2. The theorems

**Theorem A (absolute completeness, one capture).** For every finished capture `T` and every
pair `(F, m)` with `F ⊆ V_T` and `m(s) ⊆ Idx(s)`, there is a kernel sentence `S` with
`[[S]]_T = (F, m)`.

**Theorem B (relative completeness, uniform in the capture).** Assume every attribute predicate
`α ∈ P` is realized by a structural selector `σ_α` (Assumption GA, Section 4). Then for every
site query `q ∈ Q` there is a kernel sentence `tr(q)`, containing no reference to any capture,
such that for every finished capture `T` satisfying the standing assumptions,
`Sites([[tr(q)]]_T) = [[q]]_T`, with two caveats:

- (Caveat 1, complement) `¬q` uses the universal sentence `U = lift(tl.all_sites())`, which selects
  the same sites as `lift(tl.label_matches(".*"))`; it resolves to all of `V_T` exactly when
  every record in `V` has a shape, and refuses otherwise.
- (Caveat 2, unbounded reach) `Reach^{*}` is translated with a concrete hop count `N`; the translation
  is exact on every capture with `|V| ≤ N + 1`. No sentence expresses it uniformly for all
  captures, because `hops` must be an integer.

For element queries, every boolean combination of element predicates realized by producers is
expressible over a fixed site family (fiberwise completeness). Element-granularity closure of a
region (the cone of a set of elements, not of a site) is not expressible today except for
single-element seeds.

**Theorem C (necessity of the closure producers).** No sentence of `K0` (atoms and booleans, no
closure producers) expresses `Reach_fwd^{≤1}(α)`: there are two captures with identical `V`,
`attr`, `shape` and saved values, differing only in `E`, on which every `K0` sentence resolves
identically while `Reach_fwd^{≤1}(α)` differs. Moreover, no first-order formula over `(E, P)`,
however many quantifiers it uses, defines `Reach^{*}`; so reachability must be a primitive at
whatever granularity the algebra addresses.

**Theorem D (what is out of the fragment).** The following are not expressible in `K` as
capture-independent sentences: reachability restricted to paths inside or avoiding a region,
path counting, graded or soft membership, and any statement about magnitude of influence.

## 3. Proofs

### 3.1 Proof of Theorem A

1. Let `(F, m)` be given. For each `s ∈ F` define a sentence `t_s`: if `m(s) ≠ ∅`,
   `t_s = tl.units(s, m(s))`, whose value is `({s}, s ↦ m(s))`; if `m(s) = ∅`, `t_s = L_s - L_s`
   where `L_s = op_s.__selection__()`, whose value is `({s}, s ↦ Idx(s) \ Idx(s)) = ({s}, s ↦ ∅)`
   by the `-` rule (subtraction keeps the site).
2. Let `S = t_{s_1} | ... | t_{s_n}` over an enumeration of `F`. By the `|` rule the family is the
   union `F` and the masks are the per-site masks, because the `t_s` touch pairwise different
   sites. If `F = ∅`, take two different sites `v ≠ w` and `S = L_v & L_w`, whose family is
   `{v} ∩ {w} = ∅` (a capture with one site only has the empty family unreachable this way;
   `tl.between(v, v) & L_w` with a second record is the same idea).
3. Hence `[[S]]_T = (F, m)`. The sentence depends on `T` (it names `T`'s sites and coordinates),
   which is why this is "absolute, one capture" and not uniform. ∎

Example 1 in Section 5 checks this on every one of the 64 site subsets of the six-op model and
on their complements.

### 3.2 Proof of Theorem B

**Lemma B1 (whole-site closure).** Call a sentence *whole-site* (*family-clean*) when its value
is whole-site (family-clean) on every capture. The family-clean sentences contain `lift(σ)` for
every structural selector `σ`, including composite and negated ones, and are closed under `|`,
`&`, `tl.neighborhood` and `tl.between`. The whole-site sentences contain the family-clean ones
and are closed under `|`, `&`, `-`, `tl.neighborhood` and `tl.between`.

*Proof.* `lift(σ)` gives every matched site a full mask (Section 1.4). For `|` and `&`, the union
or intersection of two full masks at a shared site is full, and an unshared site under `|` keeps
its full mask. The closure producers return full masks by definition, whatever their operand.
For whole-site operands, the union, intersection or difference of two masks that are each full
or empty is again full or empty, and an unshared site keeps its own full-or-empty mask. ∎

**Lemma B2 (set reading of the booleans).** On whole-site selections (all masks full or empty),
`Sites(A | B) = Sites(A) ∪ Sites(B)`, `Sites(A & B) = Sites(A) ∩ Sites(B)`,
`Sites(A - B) = Sites(A) \ Sites(B)`.

*Proof.* At a shared site, the union of two full-or-empty masks is nonempty iff one is; the
intersection is nonempty iff both are (full ∩ full = full, anything ∩ ∅ = ∅); the difference
`m_A \ m_B` is nonempty iff `m_A` is full and `m_B` is empty. At an unshared site only the owning
side's mask is present. ∎

**The translation.** Define `tr` by induction on `q ∈ Q`. Direction map: `D(fwd) = 'downstream'`,
`D(bwd) = 'upstream'`, `D(both) = 'both'`.

```text
tr(α)                 = lift(σ_α)
tr(q1 ∪ q2)           = tr(q1) | tr(q2)
tr(q1 ∩ q2)           = tr(q1) & tr(q2)
tr(q1 \ q2)           = tr(q1) - tr(q2)
tr(¬q)                = U - tr(q),  U = lift(tl.all_sites())                [caveat 1]
tr(Reach_d^{≤k}(q))   = tl.neighborhood(tr(q), hops=k, direction=D(d))
tr(Reach_d^{*}(q))    = tl.neighborhood(tr(q), hops=N, direction=D(d))      [caveat 2]
```

Two shortcuts give family-clean results where they apply and may replace the general clauses:
`tr(¬α) = lift(~σ_α)` (predicate negation before lifting) and `tr(q \ α) = tr(q) & lift(~σ_α)`.

**Proof of Theorem B, numbered.** The induction hypothesis for a query `q` is: `tr(q)` is
whole-site, and `Sites([[tr(q)]]_T) = [[q]]_T` on every capture satisfying the standing
assumptions.

1. *Atoms.* `Sites([[lift(σ_α)]]_T) = {v : σ_α(v)} = [[α]]_T` by Assumption GA and the atom rule;
   the value is family-clean. For the shortcut `lift(~σ_α)`, the structural selector `~σ_α`
   matches exactly the records not matching `σ_α`, so `Sites = V_T \ [[α]]_T` provided every
   record has a shape (otherwise refusal, caveat 1 again).
2. *Union, intersection, difference.* By Lemma B1 the result is whole-site; by Lemma B2 and the
   induction hypothesis its `Sites` is the set union, intersection or difference of the
   operands' query values.
3. *Complement.* The universal `U` has `Sites = V_T` and full masks when every record has a shape
   (the regex `.*` matches every label; resolution refuses on a record without a shape rather
   than dropping it, caveat 1). `U - tr(q)` is whole-site by Lemma B1, and by Lemma B2
   `Sites(U - tr(q)) = V_T \ [[q]]_T = [[¬q]]_T`. Its touched family is all of `V_T`: the sites of
   `[[q]]_T` stay touched with empty masks.
4. *Bounded reach.* Let `X = tr(q)`. By the closure rule of Section 1.4 the producer seeds from
   `Sites([[X]]_T)`, which is `[[q]]_T` by the induction hypothesis, whether or not `X` is
   family-clean. The code's `_n_hop_labels` starts from the seed set, performs `k` rounds of "add
   all parents, children, or both of the current frontier", stops early when the frontier is
   empty, and returns seeds plus everything added. This is the inductive definition of
   `Reach_d^{≤k}`. The result is family-clean with `Sites = Reach_d^{≤k}([[q]]_T)`. (Checked on
   the example model for every singleton seed, every direction and every hop count from 0 to
   `|V|`.)
5. *Unbounded reach.* On a finite graph, `Reach^{*}(X) = Reach^{≤|V|-1}(X)`, since a simple path
   has at most `|V| - 1` edges. So `hops=N` with `N ≥ |V| - 1` is exact. The sentence fixes `N`
   before the capture is known, hence caveat 2.
6. *Complement and difference under a reach.* Steps 3 and 4 compose with no restriction on where
   `¬` and `\` occur. The operand `U - tr(q)` of step 3 touches all of `V_T` but carries elements
   only on `V_T \ [[q]]_T`, and a closure producer reads only the sites that carry elements, so
   `tl.neighborhood(U - tr(q), ...)` grows from `V_T \ [[q]]_T`, as `Reach(¬q)` requires. This step
   is the reason the seeding rule reads `Sites` and not `Fam`: a closure seeded from the touched
   family would grow from all of `V_T`. Example 3 in Section 5 works this case.
7. *Element level.* Over a fixed family `F`, masks form `∏_{s ∈ F} 2^{Idx(s)}`, a boolean algebra
   in which `|`, `&`, `-`, `~` are union, intersection, difference and complement (Section 1.4).
   Every element predicate realized by a producer is an atom; boolean combinations are
   translated verbatim, and restricting an element query `e` to a site query `q` is
   `tr(e) & tr(q)`. Element-granularity cones for regions have no producer today (only
   `op.receptive_field.at(unit, ...)` for single elements), so `Cone(e)` for a general element
   set is outside what ships. ∎

**Remark (conservativity).** The converse also holds for the portable kernel: every `K`-sentence
built without `tl.where` denotes a query in `Q` extended by the element predicates its producers
realize, because each atom is an attribute, coordinate or value predicate and each combinator
and closure producer maps to the corresponding `Q` construct (`tl.between(A, B)` is
`Reach_fwd^{*}(A) ∩ Reach_bwd^{*}(B)`). So, on whole-site sentences, the kernel and `Q` define
the same site sets.

### 3.3 Proof of Theorem C

**Lemma C.1 (edge-blindness of K0).** Let `T1`, `T2` be captures with the same `V`, `attr`,
`shape`, saved values and parameters, differing only in `E`. Every `K0` sentence resolves
identically on `T1` and `T2`.

*Proof.* By induction on the sentence. Each atom's resolver reads only: `attr` (the structural
selectors listed in Section 1.3; the inventory is closed, and `tl.where` is excluded from `K`),
coordinates (`tl.units`, facets, the seeded geometry of `tl.random_like`), saved values of the
resolution capture or of explicit evidence captures (value, statistical and comparative
producers; the evidence captures are the same fixed data on both sides), or basis geometry
(`tl.subspace`). None reads `E`. The combinators are pointwise in the operands' values. ∎

**Proposition C.2 (reachability sees the edges).** Take the two models

```text
M:  a = relu(x);  h = a * 2;  d = h + 1;  b = d + a
M': a = relu(x);  h = x * 2;  d = h + 1;  b = d + a
```

Both execute the same functions in the same order on tensors of the same shape, so their records
carry the same labels (`relu_1_1:1`, `mul_1_2:1`, `add_1_3:1`, `add_2_4:1`) and the same `attr`;
they differ only in whether `h` consumes `a` or `x`. With `α = {relu_1_1:1}`, `Reach_fwd^{≤1}(α)`
is `{a, h, b}` on `M` and `{a, b}` on `M'`. By Lemma C.1 no `K0` sentence separates them, so none
expresses `Reach_fwd^{≤1}(α)`. ∎

**The first-order strengthening.** Suppose a first-order formula `φ(x)` over `(E, P)` defined
`Reach_fwd^{*}(α)` on all finite DAGs. Every first-order formula is Hanf-local (Libkin, *Elements
of Finite Model Theory*, 2004, Theorem 4.24, building on Hanf 1965 and Gaifman 1982): there is a
radius `r`, depending only on `φ`, such that if two structures have the same number of points of
each `r`-neighbourhood isomorphism type, and two points `x`, `y` in them have isomorphic
`r`-neighbourhoods, then `φ(x)` holds in the first iff `φ(y)` holds in the second. Take
`n > 2r + 2` and the two structures

- `S`: two disjoint directed paths `p_0 → ... → p_{2n}` and `q_0 → ... → q_{2n}`, with `α` true
  at `p_0` only;
- `S'`: the same points, with the middle edges swapped: `p_n → q_{n+1}` and `q_n → p_{n+1}`
  replace `p_n → p_{n+1}` and `q_n → q_{n+1}`; `α` true at `p_0` only.

Every point of either structure lies on a path segment, and the swap happens farther than `r`
from every path end, so the two structures have exactly the same `r`-neighbourhood types with the
same counts, and `p_{2n}` has the same `r`-neighbourhood in both (an end segment with no
`α`-point). Yet in `S` the point `p_{2n}` is reachable from `p_0`, while in `S'` it is reachable
only from `q_0`, so `p_{2n} ∈ Reach_fwd^{*}(α)` in `S` and `p_{2n} ∉ Reach_fwd^{*}(α)` in `S'`.
This contradicts Hanf locality. Both structures are DAGs, so the argument applies to trace
graphs. This is the classical fact that transitive closure is not first-order definable (Aho and
Ullman, 1979, for the database form). Bounded reach `Reach^{≤k}` *is* first-order definable, with
quantifier rank growing with `k`, which is why the kernel needs `hops` as a parameter rather than
a fixed formula. ∎

The part proved from scratch (Lemma C.1 and Proposition C.2) is the part specific to this system:
that no shipped attribute generator leaks edge information. It holds only because the generator
inventory is closed and because the opaque `tl.where` is kept outside `K` (Section 4, Assumption
A4).

### 3.4 Proof of Theorem D

1. *Relativized reach.* "Sites on a path from `A` to `B` that avoids the region `H`" is not
   `tl.between(A, B) - H`: on the example model `between(a, b) - {h}` has sites `{a, d, b}`,
   while the only `h`-avoiding path is the direct edge, giving `{a, b}` (Example 2). It is
   expressible on one capture by iterating `X ← X | (neighborhood(X, 1, 'downstream') & R)` at
   most `|V|` times inside the allowed region `R` (Example 2), which is a sentence whose length
   grows with `|V|`, not a uniform sentence.
2. *Counting, soft membership, magnitude.* Queries such as "sites with at least three
   vertex-disjoint paths to `B`", weighted or fuzzy regions, and "how much `A` influences `B`"
   are not sets definable in `Q`, and the algebra's contract is sets: numbers are reads (tables
   keyed by site or element), and a read re-enters the algebra only through a disclosed threshold
   or rank producer (`tl.threshold`, `tl.top_k`). ∎

### 3.5 Lemma R (the honesty labels survive composition)

Each resolved site carries a relation `rel(s) ∈ {exact, upper_bound, lower_bound, unknown}`
between its mask `m(s)` and the semantic support `sem(s)` the producer meant (a receptive-field
hull is a superset of the true influence set, so it is `upper_bound`). The code transfers
relations with

```text
JOIN(exact, x) = x,  JOIN(x, x) = x,  otherwise unknown     (used for | and &)
FLIP: exact ↔ exact, upper_bound ↔ lower_bound, unknown ↔ unknown   (used for ~)
rel(A - B) = JOIN(rel(A), FLIP(rel(B)))
```

**Lemma R.** If `m_A rel_A sem_A` and `m_B rel_B sem_B` hold at a site, then the output relation
holds between the composed mask and the correspondingly composed support.

*Proof.* Union and intersection preserve `⊇` and `⊆` separately (`m_A ⊇ sem_A` and
`m_B ⊇ sem_B` give `m_A ∪ m_B ⊇ sem_A ∪ sem_B` and `m_A ∩ m_B ⊇ sem_A ∩ sem_B`; likewise for
`⊆`); mixed directions give no containment, which is `unknown`; an `exact` operand leaves the
other operand's relation as the output. Complement within `Idx(s)` reverses containment, which is
`FLIP`. Difference is intersection with a complement. ∎

The examples test checks the lemma against the code's tables over all sixteen relation pairs
with random mask and support pairs.

## 4. Assumptions and the code facts they rest on

Each item names what the proof uses and where the code guarantees it (paths relative to the
repository root).

- **GA (generator adequacy).** Every attribute predicate used in a query is a shipped structural
  selector, or is registered through the extension seam (`register_term_resolver` in
  `torchlens/_selection/selection.py`; the `__selection__` protocol). This is the hypothesis of
  Theorem B, not a fact to verify.
- **A1 (universe).** `V` is the trace's full record list (computed ops plus input and output
  records) restricted to records with a string label (`_graph_indexes` in `torchlens/receptive_field/_path.py`; `_ExecutedDag.from_trace` in
  `torchlens/_selection/graph.py`); sites are those with a tuple-of-ints `shape` (`_site_shape`
  in `torchlens/_selection/selection.py`). Multi-output functions are recorded per slot with a
  shape each (`torch.max(x, dim)` gives one record per output). Caveat 1 holds exactly when no
  record lacks a shape.
- **A2 (edges).** `E` is built from `op.parents` resolved against labels (`_graph_indexes`).
  Capture completeness, that every realized dependence is a recorded parent, is TorchLens's
  validation claim and is assumed, not proved here.
- **A3 (pass-qualified unrolling).** Graph nodes are pass-qualified labels; a cell executed twice
  yields two nodes, and upstream closure from the second pass reaches the first but downstream
  closure does not.
- **A4 (no attribute generator reads edges).** The structural selectors of Section 1.3 read
  graph-free record fields (the selector classes in `torchlens/_selection/selectors.py`; every
  internal selector kind behind spec loading reads record attributes too). Two exceptions are
  handled as follows. `tl.where(f)` hands `f` the whole record, which exposes `parents`,
  `children`, `has_output_descendant`, `min_distance_from_input` and a back-reference to the
  trace (`torchlens/_data_classes/op.py`), so `where` can compute any reachability (for example,
  `tl.where(lambda op: len(op.parents) >= 2)` resolves). Its target spec is marked
  `portable: False` (`WhereSelector` in `torchlens/_selection/selectors.py`) and it is outside
  `K`. `tl.followed_by` and `tl.preceded_by` are one-hop adjacency predicates but are capture-time
  only: the resolution door refuses them (`ensure_supported(..., lifecycle="site")` in
  `torchlens/_ir/selector_eval.py`).
- **A5 (bounded reach).** `hops` must be a non-negative `int` (`tl.neighborhood` rejects `None`,
  booleans and negatives); `_n_hop_labels` in `torchlens/_selection/graph.py` is a bounded
  breadth-first walk with early exit, so `hops=|V|` costs one traversal, not `|V|` of them.
- **A6 (between).** `tl.between` unions the descendants of all sources and the ancestors of all
  sinks before intersecting (`_between_label_set` in `torchlens/_selection/graph.py`): the "any
  source to any sink" reading.
- **A7 (seeding from the sites).** Closure producers seed only from touched sites whose mask is
  nonempty, and a partial mask seeds its whole node (`_region_labels` in
  `torchlens/_selection/graph.py`; it feeds `tl.neighborhood`, `tl.between`, `trace.between` and
  `trace.subgraph`). This is the fact behind steps 4 and 6 of Theorem B. A fully emptied operand
  gives the empty selection.
- **A8 (universal spelling).** `tl.all_sites()` is the named universal selector; it stores the
  same kind and pattern as `tl.label_matches(".*")`, matches every record and resolves to all
  sites. `tl.where(lambda op: True)` does too, but it is outside `K`.
- **A9 (closed kinds and directions).** Kinds `ACT | PARAM | EDGE` never mix
  (`selection_kind_incompatible`); forward (activation) and backward (grad-function) selector
  trees refuse to intersect or complement across the two universes
  (`selector_cross_universe_and`, `selector_cross_universe_not`; `_classify_selector_direction`
  in `torchlens/_selection/selector_base.py`). The theorems are per kind and per direction.
- **A10 (relation tables).** `_FLIP`, `_join_relation` and `_difference_relation` in
  `torchlens/_selection/provenance.py` are the tables of Lemma R.
- **A11 (the resolution door).** `trace[q]` is the only public resolution path; a set-naming
  selector returns every match as a `Selection`, a one-thing spelling (an exact `tl.label`, a
  facet) returns one record or refuses `site_spelling_not_unique`. Lifting a record with
  `__selection__` recovers the whole-site sentence, which is what `lift` means for such
  spellings.

## 5. Worked examples on a six-op model

The model has two paths from `a` to `b` (`a → h → d → b` and `a → b`). The code blocks in this
section run in order as one script.

```python
import torch

import torchlens as tl


class TwoPaths(torch.nn.Module):
    def forward(self, x):
        a = torch.relu(x)  # relu_1_1:1
        h = a * 2  # mul_1_2:1
        d = h + 1  # add_1_3:1
        b = d + a  # add_2_4:1
        return b


trace = tl.trace(TwoPaths(), torch.randn(1, 3))
inp, a, h, d, b, out = trace[tl.all_sites()].labels


def lift(*labels):
    """The whole-site sentence for the named records."""
    sentence = trace[labels[0]].__selection__()
    for label in labels[1:]:
        sentence = sentence | trace[label].__selection__()
    return sentence


def sites(selection):
    """Labels of the touched sites that carry at least one element."""
    return {label for label, row in zip(selection.labels, selection) if row.num_selected}


def family(selection):
    """Labels of every touched site, empty masks included."""
    return set(selection.labels)


U = tl.all_sites()
assert family(trace[U]) == sites(trace[U]) == {inp, a, h, d, b, out}
```

Records: `in = input_1:1`, `a`, `h`, `d`, `b`, `out = output_1:1`; edges
`in→a, a→h, h→d, d→b, a→b, b→out`. All six records have a shape (input shape `(1, 3)`), so
`V_T = V` and the universal `U = lift(tl.all_sites())` resolves to all six.

**Example 1 (a subset and its complement).** The query "the multiplication and the final add",
`{h, b}`, is `lift(h) | lift(b)`, value `({h, b}, full masks)`. Its complement `¬{h, b}` is
`U - (that)`, value: family all six, masks full on `{in, a, d, out}` and empty on `{h, b}`;
`Sites = {in, a, d, out}`. Every one of the 64 subsets of the six sites, and every complement,
resolves the same way.

```python
subset = lift(h, b)
assert sites(trace[subset]) == family(trace[subset]) == {h, b}
complement = U - subset
assert family(trace[complement]) == {inp, a, h, d, b, out}
assert sites(trace[complement]) == {inp, a, d, out}
```

**Example 2 (reach and between).** `tl.neighborhood(a, hops=1, direction='downstream')` resolves
to `{a, h, b}`; with `hops=6` to `{a, h, d, b, out}` (the unbounded reach, since
`|V| - 1 = 5 ≤ 6`). `tl.between(a, b)` resolves to `Desc(a) ∩ Anc(b) = {a, h, d, b}`. The query
"sites on an `h`-avoiding path from `a` to `b`" is `{a, b}`; the surrogate
`tl.between(a, b) - lift(h)` has `Sites = {a, d, b}`, which keeps `d` although every path through
`d` passes through `h`. The bounded iteration inside the allowed region
`R = between(a, b) - lift(h)` recovers `{a, b}` in at most `|V|` steps.

```python
assert sites(trace[tl.neighborhood(a, hops=1, direction="downstream")]) == {a, h, b}
assert sites(trace[tl.neighborhood(a, hops=6, direction="downstream")]) == {a, h, d, b, out}
assert sites(trace[tl.between(a, b)]) == {a, h, d, b}

allowed = tl.between(a, b) - lift(h)
assert sites(trace[allowed]) == {a, d, b}

region = lift(a)
for _ in range(6):
    region = region | (tl.neighborhood(region, hops=1, direction="downstream") & allowed)
assert sites(trace[region]) == {a, b}
```

**Example 3 (complement under reach).** Query: one hop downstream of the sites not downstream of
`a`. Mathematically `¬Reach_fwd^{*}(a) = {in}` and the answer is `{in, a}`. The kernel sentence
`tl.neighborhood(U - tl.neighborhood(a, hops=6, 'downstream'), hops=1, 'downstream')` resolves
to `{in, a}`. Its operand still touches all six sites, with empty masks on the five downstream of
`a`, and no boolean operator removes an element-empty site from the family (`x & x`, `~~x`,
`x | x` and `x - (x - x)` all keep the six-site family of `x = U - lift(a)`); the closure
producer seeds only from the one site that carries elements.

```python
outside = U - tl.neighborhood(a, hops=6, direction="downstream")
assert family(trace[outside]) == {inp, a, h, d, b, out}
assert sites(trace[outside]) == {inp}
answer = trace[tl.neighborhood(outside, hops=1, direction="downstream")]
assert sites(answer) == family(answer) == {inp, a}

x = U - lift(a)
for same_family in (x & x, ~~x, x | x, x - (x - x)):
    assert family(trace[same_family]) == {inp, a, h, d, b, out}
```

## 6. Out of scope and open items

- **Other sorts.** Parameter selections (`tl.param(name, mask)`) satisfy Theorem A over the
  parameter universe directly. Edge selections have a genuine universe (the trace's dataflow edge
  table) and `~` complements within it, so for kind `EDGE` complement needs no universal
  sentence. Neither sort has attribute selectors or closure producers; Theorems B and C are about
  `ACT`.
- **Tiers of influence.** Structural reachability is an upper bound on gradient support and on
  interventional influence (every dependence travels along recorded edges, Assumption A2).
  Gradient supports (`op.receptive_field.gradient(unit)`) and interventional effects
  (`fork.do(...)` followed by `tl.changed(reference=...)`) are different relations with their own
  semantics; the completeness theorems are about the structural relation `E` and say nothing
  about how much influence flows.
- **Element-granularity cones for regions.** Not shipped; see Theorem B, step 7.
- **Relativized closure** (`within=` or `avoiding=` on the closure producers). Not shipped;
  expressible per capture only (Theorem D, item 1).
- **Unbounded `hops`.** Not shipped; caveat 2 stands.
- **Subspace selections.** `tl.subspace` resolves to a support *set*, so the boolean laws of
  Section 1.4 hold for it as for any mask; the non-distributive lattice of linear subspaces never
  enters the set algebra (editing through a composed subspace is refused instead).

## References

- E. F. Codd, "Relational completeness of data base sublanguages", 1972.
- A. V. Aho and J. D. Ullman, "Universality of data retrieval languages", POPL 1979.
- W. Hanf, "Model-theoretic methods in the study of elementary logic", 1965.
- H. Gaifman, "On local and non-local properties", Logic Colloquium 1981 (1982).
- L. Libkin, *Elements of Finite Model Theory*, Springer 2004, chapter 4 (locality).
