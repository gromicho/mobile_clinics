# Predicting while optimizing: presenter narrative

**Suggested use:** approximately 20 minutes for the main presentation, plus time for the live notebook. The title page is unnumbered; headings below follow the numbered slides. Backup explanations are optional. Italic text contains presenter cues; the remaining paragraphs can be spoken directly.

## Title page — Analytics for a Better World

Today I want to explore a connection between prediction and optimisation: what happens when our decisions change the quantities we are trying to predict?

We will use mobile clinic routing as the example. This teaching companion draws on the ideas in Chapter 6 of Mayukh Ghosh's thesis, in joint work with Chintan Amrit and myself. The geography is real, but the vaccination outcomes are simulated. That gives us a controlled setting in which to examine the modelling choices and compare the resulting decisions.

## 1 — A forecast can change when the plan changes

Imagine deciding where a mobile clinic should go tomorrow. A familiar approach is to forecast attendance at each location and then find a good route through the locations with the greatest expected benefit.

But attendance may depend on the route we choose. Two nearby visits might compete for the same patients. Alternatively, several visits in an area might increase awareness and participation. In either case, the forecast belongs to a particular deployment plan.

Our question is whether we make better decisions when the optimiser can see that dependence. We will take the same fitted predictor and use it in two ways: first as a forecast made before routing, and then as a relationship inside the routing model.

## 2 — One county, twenty candidate service points

Here is Nyamira County in Kenya. We use its twenty wards as candidate service areas, with a representative point for each ward snapped to the road network.

The clinic starts in Township and can serve at most eight wards. The route is an open path: we do not require a return to the starting point. In this teaching model, the starting ward receives service and counts as one of the eight stops.

These choices make the example small enough to inspect and experiment with. For an operational study we would replace the representative points with actual service sites and identify the real collection location. That would also let us model the travel and handling requirements more faithfully.

## 3 — Data sources and what they establish

We combine boundaries, population, roads and facility locations. Each source answers a different question. GADM tells us where wards are; WorldPop gives us population estimates; OpenStreetMap supplies the road network; and the Maina database supplies historical health-facility records.

The demand mechanism comes from a simulation. We generate an unvaccinated share and specify how accessibility and nearby visits influence turnout.

There is a useful distinction here between a recorded attribute and an operational fact. A facility labelled as a hospital gives us an accessibility proxy. That label alone does not establish its current vaccine-storage capability. We should keep that distinction visible when interpreting the results.

## 4 — Facility selection changes the accessibility proxy

These two maps show why the choice of facility source matters. Both selections contain six records, but they imply different distances from wards to the nearest selected facility. In this comparison, the average is about nine kilometres using the OSM selection and about 6.8 kilometres using the Maina hospital labels.

That difference propagates into the accessibility feature and then into simulated demand. The optimisation can be internally consistent while still depending strongly on an upstream data choice. The comparison helps us see that dependence; it does not establish which records represent today's vaccine supply network.

## 5 — A bounded, set-based spillover

We represent nearby activity through an exposure measure. Each other visited ward contributes a weight that decreases with distance, and total exposure is capped at one. The interaction range controls how quickly those weights decrease.

We then multiply base demand by one plus gamma times exposure. Negative gamma gives cannibalisation: nearby visits reduce turnout at a ward. Positive gamma gives mobilisation: nearby visits increase turnout.

This is a deliberately simple mechanism. The set of selected wards determines exposure. Changing their order changes driving distance but leaves demand unchanged. We are therefore modelling a deployment effect, without specifying a chronological process in which individual patients move between stops.

## 6 — Expected service includes the capacity limit

Demand is uncertain, and a stop can serve at most 250 people. We add multiplicative noise with mean one and then apply this capacity limit.

The order matters. Expected service after a capacity limit is generally different from expected demand truncated at the capacity. If unusually high demand exceeds capacity, those extra patients cannot be served, while a low-demand day still reduces service.

Because we know the simulated noise distribution, we can calculate expected capped service analytically. We use that same expression to evaluate every route. This gives us a common measure of decision quality, without adding Monte Carlo variation to the policy comparison.

## 7 — A deliberately informative synthetic history

We generate 1500 deployment periods for training and 400 independent periods for validation. The table shows actual generated observations, including zero-exposure examples. The predictors learn capped service from noisy outcomes.

The simulation provides potential outcomes at every ward, even if a ward was not visited in that period. That is a strong information assumption, and it makes this a useful controlled teaching experiment.

With real clinic records we would normally observe less. Missing visits, capacity limits and the reasons particular routes were chosen would all affect what we can learn. Moving to real data therefore requires an identification strategy as well as a prediction algorithm.

## 8 — The same predictor, two ways to plan

This table is the central comparison. We fit a linear predictor and a small ReLU network. For each one, we compare a static policy with an embedded policy using exactly the same fitted coefficients and training history.

The static policy evaluates the predictor at zero exposure before solving the route. The embedded policy lets exposure respond to the selected wards during optimisation. Both use the same treatment of negative predictions and the same service capacity.

We also include a historical-average forecast for each ward. It is a useful simple comparator: the presence of decision dependence does not tell us in advance how much we gain from modelling it explicitly.

## 9 — Routing with decision-dependent service

The objective rewards expected service and subtracts driving cost. Here, one kilometre costs one dose-equivalent. That coefficient expresses a trade-off chosen for the example; it is not an estimated programme valuation.

The binary variables select stops and travel arcs. Service must be zero at unvisited wards and cannot exceed capacity. In the embedded model, the predictor's output changes with the selected stops, so the optimiser must consider their joint effect on service and travel.

We also solve a benchmark that knows the simulator's expected service function. Its approximation has a controlled error, which we add to the solver's bound. The full routing and predictor formulations are in the backup slides.

## 10 — Environment for these results

Before looking at results, this slide records the machine and software used to produce the displayed tables. The runs used two solver threads, with model-building and solving times recorded separately.

The point is reproducibility. These are teaching results from one machine, with cached geographic inputs and uncontrolled background load. The notebook reports solver status and gaps, so a time-limited run can be distinguished from a completed optimisation. We should not turn a short demonstration into a claim about performance on every machine.

## 11 — Cannibalisation: doses, distance and objective

*Point first to the two MLP rows, then to the linear and historical rows.*

With negative spillover, the static neural-network policy serves about 1541 expected doses and drives 73.1 kilometres. Embedding that same network changes the plan to about 1614.7 doses and 78.9 kilometres. The objective rises from 1467.9 to 1535.8. We gain service and also do more driving.

Now look at the other rows. The static linear policy, the embedded linear policy and the historical mean achieve the same displayed expected objective as the benchmark route. The benchmark places the optimum between 1535.83 and 1536.23.

So embedding helps this neural-network comparison, but it is not necessary to obtain a very good decision in this particular instance. We should explain the pattern the experiment produces, rather than assume that the most elaborate policy must win.

## 12 — Mobilisation: a benefit is not guaranteed

With positive spillover, embedding improves the objective for both fitted predictors in this run. The linear comparison rises from about 1820.8 to 1827.3; the neural-network comparison rises from 1809.5 to 1822.9.

Look carefully at what produces those improvements. For the neural network, the embedded route serves slightly fewer expected doses but drives substantially less. It performs better under our weighted objective. That is why doses, distance and their combination all belong in the table.

The historical mean reaches about 1828.0, above both embedded predictors here. The benchmark route reaches about 1828.7, with an upper bound of 1829.18. Explicitly representing decision dependence can help, but its value also depends on how well the fitted relationship represents service in the decisions the optimiser considers.

## 13 — The selected routes

The maps turn those numbers into geographic decisions. Compare the static and embedded neural-network routes within each scenario, then compare them with the benchmark.

The optimiser can change the set of visited wards, the order of travel, or both. In our simulator, selecting different wards changes service through exposure; changing the order changes travel cost. Looking at the maps helps us connect those two mechanisms to the aggregate results.

These are open paths, even when a map looks roughly circular. There is no required final journey back to Township in the objective.

## 14 — Sensitivity to the assumed interaction

So far we have examined two choices of gamma. The sweep asks how the comparisons change across a wider range of assumed interactions.

At strongly negative spillover the policies separate substantially. Around zero and for positive spillover they are much closer on this scale. We use matched training histories at each strength, and evaluate every route with the same expected-service calculation. The narrow benchmark band represents numerical optimisation and interpolation uncertainty.

That band is not a confidence interval for the real-world effect. Gamma itself is an assumption in this experiment. This plot tells us how the model behaves when we change that assumption.

## 15 — Live demo: a prediction before each run

*Pause for the audience's prediction before changing a parameter. Use one experiment if time is short.*

First, suppose we shorten the interaction range. Which wards will still affect each other, and how might that change our preferred stops? Think separately about doses, kilometres and the weighted objective.

Next, change the maximum number of stops. We are allowing more stops, not requiring them. For the exact benchmark, expanding the feasible set cannot reduce the optimal weighted objective, although the selected route and its distance can change.

Finally, reduce the neural network from eight hidden units to two while keeping its training history fixed. This changes the fitted approximation and the optimisation formulation. A smaller predictor may be easier to embed, but the relevant question is how its errors affect the decisions.

The notebook also repeats the comparison across three training-history seeds. Those repetitions help distinguish a pattern in this example from a result tied to one generated history.

## 16 — What would make this an operational study?

To use this approach operationally, we would need actual service and collection sites, verified logistics, and the relevant travel and time constraints. We would also need to understand what historical records reveal about turnout, capacity and spillovers, and validate decisions in the setting where they will be used.

There may be programme objectives that this example does not represent, such as equity, continuity of service or reaching particular groups. Those belong in the problem definition and evaluation.

The idea I would like you to take away is that a prediction can be part of the decision model. When a decision changes what will happen, we can represent that relationship explicitly. Whether doing so improves the decision is an empirical and modelling question, which we should answer with matched comparisons, credible data and a clearly stated objective.

## Backup 17 — Model notation and inputs

The model has one candidate service point per ward and a directed arc between every pair of different wards. An arc's length is a shortest-road-path distance, rather than a straight-line distance. The input features are log population, a simulated unvaccinated share and a normalised distance to a hospital proxy.

The binary variables determine which wards receive service and which arcs connect them. Service variables are continuous. Capacity is a limit at each stop; there is no constraint here representing a fixed stock of vaccines carried by the vehicle. The latter would be an additional modelling choice.

## Backup 18 — Complete routing and service formulation

Read the formulation from the top. The objective rewards modelled service and subtracts travel. The next line selects the depot, prevents returning into it, and limits the number of selected wards.

Every selected non-depot ward has one incoming arc. A selected ward has at most one outgoing arc, and the total number of arcs is the number of selected wards minus one. The remaining danger is a disconnected cycle, which the subtour inequalities eliminate.

For a selected cycle, the number of internal arcs would equal the number of its selected nodes, violating the displayed inequality. The callback adds these inequalities when an integer candidate contains a disconnected cycle. Along with the degree and arc-count constraints, they enforce a single open path starting at the depot. Selecting only the depot is feasible and requires no travel.

The final service bounds link the route to the chosen prediction rule. Because service has a positive objective coefficient and no other use, it rises to its permitted value at selected wards.

## Backup 19 — Exposure and policy-specific service

Travel remains directed. For the interaction weights we deliberately use the average of the two directed distances, making the assumed geographic influence symmetric. A ward does not contribute to its own exposure.

The embedded formulation computes the weighted sum of other visits and caps it at one. The predictor sees that decision-dependent exposure. The static formulation instead evaluates the same predictor at zero exposure in advance. The historical baseline uses each ward's average simulated capped service.

Taking the maximum with zero handles negative predictor outputs consistently. Combining that bound with capacity gives the same zero-to-capacity clipping in static and embedded policies. The minimum and maximum relations are exact solver constraints; they are not smooth approximations.

## Backup 20 — The fitted predictor inside the model

Embedding includes the preprocessing. We first apply the means and scales estimated during training to all four inputs, including exposure. The linear model is then an affine equation.

For the default neural network, eight affine expressions feed eight ReLU units. Their nonnegative outputs feed a final affine prediction, which is clipped below at zero. All weights, intercepts and scaling constants are fixed before routing begins. We are optimising the deployment with a trained network, not training a network inside the routing solver.

Gurobi's general constraints represent the ReLU and exposure relations exactly through mixed-integer machinery. These equations describe the default one-hidden-layer network used in the displayed experiments; the software also permits other configured layer sizes.

## Backup 21 — Synthetic service and its expectation

The simulated service at a ward is the smaller of capacity and noisy demand. The lognormal parameterisation makes the multiplicative noise have mean one. The expression involving the normal distribution function integrates over both the days below capacity and the days on which capacity binds.

This distinguishes three quantities: a noisy observation used for fitting, a predictor's estimate used by a policy, and the simulator's exact expected service used for evaluation. The benchmark has access to that last quantity. Its information advantage is intentional, which is why it is a reference within the simulation rather than an implementable policy learned from ordinary clinic records.

## Backup 22 — Simulator benchmark and route evaluation

Expected capped service is concave as a function of exposure. We place a grid on the exposure interval and join adjacent function values with secants. For a concave function, that piecewise-linear interpolation lies below the true function.

The minimum of the extended secant lines gives that interpolation over the grid domain. We can therefore impose one upper bound on service for each line, keeping the routing model linear apart from its discrete and general-constraint components.

After selecting a route, we evaluate its service with the exact expectation. We do this for every policy, including the benchmark. A policy's internal predicted objective and its actual expected objective under the simulator can differ; the presentation compares the latter.

## Backup 23 — A bound on benchmark interpolation

The curvature bound controls the largest possible error between a secant and the expected-service function. We choose the grid so that the error at any one ward is at most the total tolerance divided by the maximum number of stops. At most that many wards can contribute service, so their total error is at most 0.5 dose-equivalent in this example.

The exact expected value of a feasible route is a lower bound on the simulator optimum. The solver's upper bound on the interpolated model, plus the interpolation allowance, is an upper bound on that optimum. The interval shown in the results combines these two statements.

This is a numerical guarantee for the specified simulator and routing assumptions. It does not quantify uncertainty about real patient behaviour or the suitability of the data.

---

**Evidence for the script:** the displayed tables come from `results/scenario_a.json` and `results/scenario_b.json`; sensitivity statements use `results/sweep.json` and `results/history_sensitivity.json`. The formulas follow `clinic_routing.py` and `clinic_data.py`. Dataset sources and terms are in [DATA_SOURCES.md](../DATA_SOURCES.md). The [presentation](intro.pdf) and this narrative refer to the same committed numerical snapshot; presenting different notebook runs may require updating the numerical passages above.
