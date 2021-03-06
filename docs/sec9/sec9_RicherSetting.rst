9 A richer economic setting
===========================

While our main analysis has so far focused on a stylized model of the
coupled climate-economy dynamics, our framework is also well-suited for
exploring richer environments. Important modeling extensions include
additional macroeconomic components that allow for adaptation,
technological change, and multiple sectors. We have left off the table
discussions of technological innovation that could support the
transition from dirty to clean capital and the reallocating production
from technologies that are vulnerable to climate change to ones that are
more resilient. Modeling extensions that allow us to address a broader
range of economic responses are important both for assessing the role of
policy and for accommodating market responses. Rather than constructing
an all purpose integrated assessment model, we describe a two-capital
model setup that allows us to confront some of the gaps left with our
illustrative model.

| We take as starting point the two capital, AK specification of
  :cite:t:`EberlyWang:2012` that includes adjustment costs and
  hence sluggish reallocation. The resulting technologies are stochastic
  subject to Brownian increment shocks and are used to output that is
  divided between investment and consumption. It is a two capital
  extension of the undamaged consumption/investment model that we have
  used so far. Although their model has stochastic growth, many
  parameter configurations imply long-term stationary behavior of the
  relative capital stocks. The ratio of the capital stock becomes a
  featured state variable for their model.
| We start with two capital stocks that evolve as:

.. math::


   d K_{j,t} =  K_{j,t}   \left[ \mu_{j,k} (Z_t) \cdot dt + \left({\frac {I_{j,t}}{K_{j,t}}} \right)dt - {\frac { \kappa_j} 2} \left( {\frac {I_{j,t}} {K_{j,t}}} \right)^2 dt
   + \sigma_{j,k}(Z_t) \cdot dW_t^k \right]

and :math:`K_{j,t}` and :math:`I_{j,t}` are technology specific capital
and investment for :math:`i=1,2`. The output of each production is
proportional:

.. math::


   I_{j,t} + C_{j,t} = \alpha_j K_{j,t}

For our first application, the planner has an instantaneous utility
function that can be represented as:

.. math::


   \log \left[ \left( C_{1,t}\right)^{1-\eta} \left( {\mathcal E}_{1,t} \right)^\eta  + \left( C_{2,t}\right)^{1-\eta} \left( {\mathcal E}_{2,t} \right)^\eta\right] 
    - (1 -\eta) \log N_t 

| where the two sources of emissions contribute to damages
  differentially. Here, we have in mind coal and oil where coal use
  generates more emissions, and therefore leads to more climate damages
  than oil.
| In the absence of climate change considerations, both technologies are
  attractive sources of production. Taking account of climate change, a
  planner will shift production to the cleaner of the two technologies.
  This model opens the door to considering uncertainty and policy as it
  pertains to taxing or restraining coal production.

Next, we change the configuration to consider clean versus dirty
technologies. To do so, we drop emissions from the second technology and
replace it by an input available in fixed supply, :math:`\tau`. The
resulting instantaneous utility function is of the form:

.. math::


   \log \left[ \left( C_{1,t}\right)^{1-\eta} \left( {\mathcal E}_{1,t} \right)^\eta  +  \left( C_{2,t}\right)^{1-\eta} \tau^\eta\right] 
    - (1 -\eta) \log N_t 

We suppose initially that there is a productivity advantage for the
dirty technology, but that there is a third investment opportunity
devoted to research and development (R&D) that can induce improvements
in the production from the green sector. The outcome of accumulated R&D
investment is the eventual stochastic arrival of a more productive green
technology where the arrival is modeled as a jump process with intensity
linked to the current stock of R&D. This setting is reminiscent of the
framework explored by :cite:t:`Acemogluetal:2016`, though
their focus is on the dynamics of innovation abstracting from capital
reallocation and concerns about uncertainty. This second application
opens the door to studying potential subsidies for R&D in the face of
broadly conceived uncertainty.

As a third application, we suppose that one of the two production
technologies is vulnerable to climate change while the other one is not.
Consumption and investment from the two technologies are treated as
perfect substitutes. We add an additional source of damages that might
be triggered in the future but for only the first technology. Thus, the
first technology is more vulnerable to climate change, as there is a
productivity decline with an uncertain arrival time that becomes more
likely as temperature increases. Given the sluggishness in reallocating
capital, there can be non-trivial transitional consequences to this drop
in productivity. This setting allows us to explore the role of policy
for monitoring or limiting this vulnerability in the presence of damage
uncertainties. See :cite:t:`Fantetal:2020` and
:cite:t:`Klingetal:2021` for recent research quantifying
vulnerability of productive activities to climate change and the
potential need for adaptation. In a different vein, valuable
quantitative research has been done building probabilistic models of
climate tipping points and assessing their risk consequences. See, for
instance, :cite:t:`Lentonetal:2008` and
:cite:t:`Caietal:2015`. A model such as the one we described
opens the door to the study of heterogeneous vulnerability to tipping
point uncertainty conceived of more broadly than is typical in risk
analyses.

While other studies have explored policy challenges introduced by
related models, we will study the broad consequences of uncertainty in
such environments. In each case, we can include the analogous
uncertainty concerns through model misspecification and ambiguity as
before. By enriching the dynamics, however, we also open the door to new
channels by which uncertainty comes into play and to alternative policy
levers worthy of exploration.
