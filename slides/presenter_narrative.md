# Predicting while optimizing: presenter narrative

**Suggested use:** approximately 25–30 minutes for the main presentation with the expanded formula explanations, plus time for the live notebook. For a shorter talk, omit the worked examples. The title page is unnumbered; headings below follow the numbered slides. Backup explanations are optional. Italic text contains presenter cues; the remaining paragraphs can be spoken directly. Equations in these notes support the explanation even where they are not printed on the main slide.

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

We use distance to the nearest hospital as a simple measure of healthcare accessibility. The dataset does not record vaccine-storage capacity, so an operational application would need to verify where vaccines can actually be collected.

## 4 — Facility selection changes the accessibility proxy

These two maps show why the choice of facility source matters. Both selections contain six records, but they imply different distances from wards to the nearest selected facility. In this comparison, the average is about nine kilometres using the OSM selection and about 6.8 kilometres using the Maina hospital labels.

In the simulation, greater distance to a hospital increases base demand for a mobile clinic. The assumption is that wards with poorer access to existing facilities have more unmet need. We represent that assumption with

$$
a_i=\frac{\rho_i}{\max_k\rho_k},\qquad
b_i=0.02\,P_i u_i(0.5+a_i).
$$

Here, rho is the road distance from the ward's representative road node to its nearest selected hospital. Dividing by the largest such distance in the county gives a relative inaccessibility score, a, between zero and one. A larger score means poorer access. P is the ward population and u is its simulated unvaccinated share, so their product is the simulated number of unvaccinated people.

The factor 0.02 sets the scale of demand for a visit. The factor 0.5 plus a adjusts that scale for inaccessibility: it ranges from 0.5 at zero hospital distance to 1.5 at the greatest hospital distance. These coefficients are chosen for the teaching simulation; they are not estimated attendance rates. The resulting b is base mean demand before nearby visits, random variation and the service limit are applied.

*Optional worked example; the numbers are illustrative, not an additional ward observation.*

Take a ward with 20,000 people and an unvaccinated share of 40 percent. Their product is 8000, and two percent of that is 160. If the inaccessibility score is 0.25, base demand is 160 times 0.75, or 120 people. If the score is 0.75, with the other inputs unchanged, base demand is 160 times 1.25, or 200 people. That is the assumed effect of poorer hospital access.

Changing the facility source changes the distances and can therefore change these scores and base demands. Because we normalise by the county maximum, the relative pattern matters: a proportional increase in every distance would leave the scores unchanged. The lower average distance in the Maina map does not by itself imply lower demand in every ward. The displayed routing experiments use the Maina hospital selection; the two maps illustrate an input choice rather than a complete routing comparison between the two sources.

Hospital distance is fixed while the optimiser chooses a route. On the next slide, we introduce a second distance relationship: proximity to other mobile-clinic visits, whose effect depends on which wards we select.

## 5 — A bounded, set-based spillover

We represent nearby mobile-clinic activity through an exposure measure:

$$
\bar d_{ij}=\frac{d_{ij}+d_{ji}}{2},\qquad
w_{ij}=e^{-\bar d_{ij}/R}\ (i\ne j),\qquad
E_i(y)=\min\left\{1,\sum_{j\ne i}w_{ij}y_j\right\}.
$$

Here, y sub j is one if ward j is selected for service and zero otherwise. A selected ward contributes to exposure at other wards, with a weight that decreases exponentially with their road distance. We average the two directed distances for this interaction, although the driving cost still uses the actual direction of travel. A ward does not contribute to its own exposure.

The parameter R controls how far the influence extends. With R equal to ten kilometres, a visit ten kilometres away contributes about 0.37; one twenty kilometres away contributes about 0.14. This is a gradual decay, not a cutoff at ten kilometres. Contributions from multiple selected wards add up, and the cap at one prevents exposure from increasing without limit. For example, two visits each five kilometres away contribute about 0.61 each, giving a capped exposure of one.

We then apply the same exposure measure in two alternative scenarios:

$$
\mu_i(y)=b_i\bigl(1+\gamma E_i(y)\bigr).
$$

For **cannibalisation**, gamma is negative. Nearby visits reduce mean demand at the ward. With gamma equal to minus 0.5, the multiplier falls from one at zero exposure to 0.5 at full exposure. This represents competition between nearby service opportunities.

For **mobilisation**, gamma is positive. Nearby visits increase mean demand. With gamma equal to plus 0.5, the multiplier rises from one to 1.5. This represents a possible increase in participation through awareness or mobilisation around a cluster of visits.

*Optional worked example connecting the two scenarios.*

If base demand is 200 and exposure is 0.6, cannibalisation gives 200 times 0.7, or 140 people in mean demand. Mobilisation gives 200 times 1.3, or 260. These are demand values before applying noise and the 250-person service limit. At zero exposure both scenarios return to base demand.

We run these as separate scenarios, each with one common gamma across the county. The simulation does not model both mechanisms simultaneously or estimate their strength from patient records. Awareness and competition motivate the signs; the mathematical mechanism is the exposure multiplier.

This is a deliberately simple mechanism. The set of selected wards determines exposure. Changing their order changes driving distance but leaves demand unchanged. We are therefore modelling a deployment effect, without specifying a chronological process in which individual patients move between stops.

## 6 — Expected service includes the capacity limit

Demand is uncertain, and a stop can serve at most 250 people. We add multiplicative noise with mean one and then apply this capacity limit.

Putting the pieces together, potential service at ward i is

$$
S_i(y)=\min\left\{250,\;
\underbrace{0.02P_i u_i(0.5+a_i)}_{\text{base demand}}
\underbrace{(1+\gamma E_i(y))}_{\text{effect of nearby visits}}
\underbrace{\epsilon_i}_{\text{random variation}}\right\}.
$$

Read that expression from left to right inside the capacity limit: population and hospital access establish base demand; the chosen deployment changes it through exposure; random variation changes turnout on the day; and capacity limits how many people can be served. We count this potential service in the route's total only when the ward is selected.

The noise is lognormal, which keeps demand positive. Its logarithm has mean minus sigma squared over two and standard deviation sigma, with sigma set to 0.15. That adjustment makes the noise multiplier itself have mean one, so it does not systematically inflate the demand scale.

In the mobilisation example, mean demand rose to 260, but the clinic cannot serve 260 people at a stop. Without noise it would serve 250. With noise, some days have demand below 250, so expected service is below 250. Increased demand can therefore produce a much smaller increase in expected service when capacity is already tight.

The order matters. Expected service after a capacity limit is generally different from expected demand truncated at the capacity. If unusually high demand exceeds capacity, those extra patients cannot be served, while a low-demand day still reduces service.

Because we know the simulated noise distribution, we can calculate expected capped service analytically. We use that same expression to evaluate every route. This gives us a common measure of decision quality, without adding Monte Carlo variation to the policy comparison.

## 7 — A deliberately informative synthetic history

We generate 1500 deployment periods for training and 400 independent periods for validation. The table shows actual generated observations, including zero-exposure examples. The predictors learn capped service from noisy outcomes.

The simulation provides potential outcomes at every ward, even if a ward was not visited in that period. That is a strong information assumption, and it makes this a useful controlled teaching experiment.

For each generated deployment we calculate exposure from its selected wards, apply the chosen gamma, draw the noise and cap service. We fit separate predictors for the cannibalisation and mobilisation scenarios. Their inputs are log population, the unvaccinated share, inaccessibility and exposure; their target is the simulated capped service. The predictors must learn that relationship from the examples. They are not given the simulator's demand formula as a constraint.

With real clinic records we would normally observe less. Missing visits, capacity limits and the reasons particular routes were chosen would all affect what we can learn. Moving to real data therefore requires an identification strategy as well as a prediction algorithm.

## 8 — The same predictor, two ways to plan

This table is the central comparison. We fit a linear predictor and a small ReLU network. For each one, we compare a static policy with an embedded policy using exactly the same fitted coefficients and training history.

The static policy evaluates the predictor at zero exposure before solving the route. The embedded policy lets exposure respond to the selected wards during optimisation. Both use the same treatment of negative predictions and the same service capacity.

This distinction is particularly useful for understanding the two effects. Under cannibalisation, zero exposure describes the absence of competing visits. Under mobilisation, it describes the absence of an attendance boost from nearby visits. The static policy keeps those forecasts fixed even when its route selects nearby wards. The embedded policy updates exposure within the model. Its response is whatever the fitted predictor has learned, which may differ from the simulator's exact response.

Embedding therefore does not mean inserting the known gamma formula into the learned policy. The learned policy contains the fitted linear model or neural network. Only the simulator benchmark uses the known expected-service relationship directly.

It helps to separate what we specify from what we learn. We specify how selected wards produce exposure: the distance weights, interaction range and exposure cap are modelling assumptions. We learn how population, the unvaccinated share, hospital inaccessibility and that exposure jointly predict capped service. We fit one shared predictor across the wards for each scenario, then freeze its coefficients before routing.

Hospital inaccessibility enters as a fixed input for each ward. Exposure enters as an input whose value is determined by the optimisation decisions. Thus the learned model can represent both the fixed accessibility effect and the deployment-dependent spillover effect, but it approximates their combined effect from training observations. It does not need to reproduce each factor of the simulator formula explicitly.

We also include a historical-average forecast for each ward. It is a useful simple comparator: the presence of decision dependence does not tell us in advance how much we gain from modelling it explicitly.

## 9 — Routing with decision-dependent service

The objective rewards expected service and subtracts driving cost. Here, one kilometre costs one dose-equivalent. That coefficient expresses a trade-off chosen for the example; it is not an estimated programme valuation.

The binary variables select stops and travel arcs. Service must be zero at unvisited wards and cannot exceed capacity. In the embedded model, the predictor's output changes with the selected stops, so the optimiser must consider their joint effect on service and travel.

The connection is a chain of constraints:

$$
y\ \longrightarrow\ E_i(y)\ \longrightarrow\
q_i=g(\log P_i,u_i,a_i,E_i)\ \longrightarrow\
\widehat h_i=\max\{0,q_i\},\qquad
0\le z_i\le Cy_i,\quad z_i\le\widehat h_i.
$$

The prediction q is a solver variable, but the optimiser cannot choose it freely: the fitted predictor's equations determine its value from the inputs. Selecting another ward can change exposure, which changes the permitted prediction and hence the service we can claim in the objective. The optimiser considers that service effect together with the travel consequences.

For a selected ward, the objective pushes service up to the smaller of capacity and its nonnegative prediction. For an unselected ward, service is zero. For example, a prediction of 180 permits 180 doses at a visited ward; a prediction of 270 permits only 250; and a negative prediction permits zero. These are illustrations of the constraints, not additional fitted results.

This is why we call them learned constraints: the coefficients determining the service bound come from a fitted model. We are inserting the predictor's equations into the optimisation problem, so the solver can reason about their consequences while choosing the route. The static version computes those bounds once at zero exposure and keeps them constant.

There are now three roles for distance. Distance to hospitals helps set fixed base demand. Distance between selected wards determines exposure and its effect on demand. Distance along the chosen travel arcs incurs a cost. Serving a poorly connected ward can therefore offer higher simulated need while also requiring more driving; selecting nearby wards can shorten travel while changing their turnout. The objective weighs these consequences together.

We also solve a benchmark that knows the simulator's expected service function. Its approximation has a controlled error, which we add to the solver's bound. The full routing and predictor formulations are in the backup slides.

## 10 — Environment for these results

Before looking at results, this slide records the machine and software used to produce the displayed tables. The runs used two solver threads, with model-building and solving times recorded separately.

The point is reproducibility. These are teaching results from one machine, with cached geographic inputs and uncontrolled background load. The notebook reports solver status and gaps, so a time-limited run can be distinguished from a completed optimisation. We should not turn a short demonstration into a claim about performance on every machine.

## 11 — Cannibalisation: doses, distance and objective

*Point first to the two MLP rows, then to the linear and historical rows.*

Here gamma is minus 0.5. Increasing exposure reduces mean demand, by at most half of base demand at full exposure. This gives the model an incentive to avoid overlapping service opportunities, but that incentive competes with travel cost, differences in base demand and the capacity limit. It does not imply that the best route always selects the most widely separated wards.

With negative spillover, the static neural-network policy serves about 1541 expected doses and drives 73.1 kilometres. Embedding that same network changes the plan to about 1614.7 doses and 78.9 kilometres. The objective rises from 1467.9 to 1535.8. We gain service and also do more driving.

Now look at the other rows. The static linear policy, the embedded linear policy and the historical mean achieve the same displayed expected objective as the benchmark route. The benchmark places the optimum between 1535.83 and 1536.23.

So embedding helps this neural-network comparison, but it is not necessary to obtain a very good decision in this particular instance. We should explain the pattern the experiment produces, rather than assume that the most elaborate policy must win.

## 12 — Mobilisation: a benefit is not guaranteed

Here gamma is plus 0.5. Exposure increases mean demand, by up to half of base demand at full exposure. Nearby visits can reinforce one another, and a geographically compact selection may also save travel. However, the exposure cap and the 250-dose service cap limit the benefit of additional nearby visits. This model does not reward clustering indefinitely.

With positive spillover, embedding improves the objective for both fitted predictors in this run. The linear comparison rises from about 1820.8 to 1827.3; the neural-network comparison rises from 1809.5 to 1822.9.

Look carefully at what produces those improvements. For the neural network, the embedded route serves slightly fewer expected doses but drives substantially less. It performs better under our weighted objective. That is why doses, distance and their combination all belong in the table.

The historical mean reaches about 1828.0, above both embedded predictors here. The benchmark route reaches about 1828.7, with an upper bound of 1829.18. Explicitly representing decision dependence can help, but its value also depends on how well the fitted relationship represents service in the decisions the optimiser considers.

## 13 — The selected routes

The maps turn those numbers into geographic decisions. Compare the static and embedded neural-network routes within each scenario, then compare them with the benchmark.

The optimiser can change the set of visited wards, the order of travel, or both. In our simulator, selecting different wards changes service through exposure; changing the order changes travel cost. Looking at the maps helps us connect those two mechanisms to the aggregate results.

These are open paths, even when a map looks roughly circular. There is no required final journey back to Township in the objective.

## 14 — Sensitivity to the assumed interaction

So far we have examined two choices of gamma. The sweep asks how the comparisons change across a wider range of assumed interactions.

Moving left from zero makes the exposure penalty stronger; moving right makes the exposure benefit stronger. At gamma equal to zero, mean demand equals base demand whatever wards are selected, so the simulator has no spillover effect. Any remaining sensitivity of a fitted predictor to exposure at that setting comes from fitting error rather than a true interaction in the simulator. The sweep holds the interaction range fixed: gamma changes the sign and strength, not the geographic decay of exposure.

At strongly negative spillover the policies separate substantially. Around zero and for positive spillover they are much closer on this scale. We use matched training histories at each strength, and evaluate every route with the same expected-service calculation. The narrow benchmark band represents numerical optimisation and interpolation uncertainty.

That band is not a confidence interval for the real-world effect. Gamma itself is an assumption in this experiment. This plot tells us how the model behaves when we change that assumption.

## 15 — Live demo: a prediction before each run

*Pause for the audience's prediction before changing a parameter. Use one experiment if time is short.*

First, suppose we shorten the interaction range. Which wards will still affect each other, and how might that change our preferred stops? Think separately about doses, kilometres and the weighted objective.

For the same selected wards, reducing R decreases the distance weights and can reduce exposure. At a distance of ten kilometres, the contribution is about 0.14 when R is five, 0.37 when R is ten, and 0.61 when R is twenty. Thus a shorter range can weaken the cannibalisation penalty in the negative scenario and the mobilisation benefit in the positive scenario. If the weighted sum still reaches the exposure cap, that ward's exposure stays at one. These statements hold for a fixed deployment; after reoptimisation, the route itself may change. Hospital inaccessibility and base demand stay fixed during this experiment.

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

To see exactly where the effects enter, write the input and its standardisation as

$$
v_i=(\log P_i,u_i,a_i,E_i),\qquad
t_{i\ell}=\frac{v_{i\ell}-m_\ell}{s_\ell},\quad \ell=1,\ldots,4.
$$

The fitted means m and scales s are constants. The first three input values are fixed for each ward; only the fourth changes with the selected deployment. For linear regression the prediction equation is

$$
q_i=\beta_0+\beta_1t_{i1}+\beta_2t_{i2}
              +\beta_3t_{i3}+\beta_4t_{i4}.
$$

The third coefficient represents an additive association with hospital inaccessibility, holding other inputs fixed. The fourth represents an additive association with exposure. Before output clipping, a change in exposure changes the prediction at the constant rate beta four divided by the fourth input scale. This linear model cannot reproduce all the interactions and capacity effects of the simulator, which is one reason to examine a more flexible predictor.

For the neural network the corresponding equations are

$$
r_{ih}=\max\left\{0,c_h+\sum_{\ell=1}^4B_{h\ell}t_{i\ell}\right\},
\quad h=1,\ldots,8,\qquad
q_i=c_0+\sum_{h=1}^8\alpha_h r_{ih}.
$$

Each hidden unit combines all four inputs before applying its ReLU. As exposure changes, some units can switch between zero and their affine expression. The resulting prediction is piecewise linear, and its response to exposure can differ across wards because their fixed inputs differ. The equations use the same learned coefficients for every ward.

In the implementation, `add_predictor_constr(model, predictor, inputs, outputs)` inserts the fitted scikit-learn pipeline, including its scaler, into Gurobi. We fix each ward's first three input variables to its observed or simulated feature values and equate its fourth input variable to exposure. Separate exact min constraints define capped exposure, max constraints impose nonnegative predictions, and the service inequalities impose selection and capacity. The library handles the predictor equations and the solver handles their mixed-integer representation.

Neither predictor is constrained to be monotone in exposure. Training on cannibalisation data may teach a decreasing response and training on mobilisation data may teach an increasing response, but those signs are not guaranteed everywhere. The optimiser sees the fitted response, including its errors. We fit by prediction error before routing; we do not train the coefficients to maximise route quality. That is why held-out prediction checks and evaluation of the final decisions both matter.

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
