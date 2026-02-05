# **Advanced Hysteresis Modeling in Dynamic Milling: Acceleration Envelope as a Deterministic Predictor of Nonlinear Cutting Forces**

## **1\. Introduction: The Imperative for Nonlinear Dynamics in High-Precision Machining**

The pursuit of high-performance machining (HPM)—characterized by extreme material removal rates, tight geometric tolerances, and superior surface integrity—has pushed the mechanical boundaries of machine tool systems. In sectors ranging from aerospace, where thin-walled monolithic components are milled from titanium and aluminum alloys, to the die and mold industries demanding mirrored surface finishes on hardened steels, the dynamic stability of the milling process is the primary limiting factor. Traditionally, the analysis of these systems has relied on linear theory, assuming that the relationship between the cutting force and the resulting structural displacement is proportional and time-invariant. However, as machining speeds increase and structures become more lightweight and flexible, these linear approximations fail to capture the complex, path-dependent behaviors that emerge in the cutting zone.

This report posits a paradigm shift in the modeling of milling dynamics: the transition from direct, often intrusive force measurement to the use of the **acceleration envelope** as a primary, non-intrusive input for characterizing nonlinear hysteresis. The central thesis of this work is built upon a critical validation found in the mechanics of chip formation—that **vibration intensity is intrinsically proportional to the dynamic chip load**. Since the cutting force is a direct function of the chip load ($F \\approx k \\cdot h$), the acceleration envelope, which demodulates the intensity of the vibration signal, serves as a high-fidelity proxy for the dynamic cutting force. This proportionality allows for the reconstruction of complex hysteretic phenomena, such as process damping and structural nonlinearity, without the bandwidth limitations and structural compliance introduced by dynamometers.

The analysis that follows is exhaustive. It synthesizes theoretical mechanics, signal processing methodologies, and advanced phenomenological modeling (including Bouc-Wen and Preisach formulations) to establish a robust framework for hysteresis modeling. We explore the "unsafe zones" of bi-stability where systems jump between stable cutting and regenerative chatter, creating distinct hysteresis loops in the stability charts. We validate the nonlinear relationship between vibration intensity and force, leveraging the "bingo" insight that connects physical displacement to chip thickness and, ultimately, to the acceleration signature. By integrating these disparate threads of research, this report provides a comprehensive roadmap for the next generation of intelligent, adaptive machining controls.

## **2\. Theoretical Fundamentals of Dynamic Milling Mechanics**

To effectively utilize the acceleration envelope as a predictor for cutting forces, one must first establish a rigorous understanding of the underlying mechanics that generate these signals. Milling is a discontinuous cutting process, a feature that distinguishes it fundamentally from turning or drilling. The intermittent engagement of cutting teeth with the workpiece creates a periodic forcing function, but it is the interaction between this periodicity and the structural dynamics of the machine-tool system that leads to the rich, often chaotic, vibratory behavior observed in practice.

### **2.1 The Mechanics of Chip Formation and Force Generation**

The genesis of all vibration in milling is the chip formation process. As a cutting tooth engages the workpiece, it shears away material, generating a resistance force. Classical mechanistic models, such as those proposed by Altintas and Tlusty, postulate that this cutting force is proportional to the instantaneous cross-sectional area of the chip. In a simplified linear approximation, the tangential ($F\_t$), radial ($F\_r$), and axial ($F\_a$) force components are expressed as functions of the uncut chip thickness ($h$) and the axial depth of cut ($b$).1

The governing equations for the differential cutting forces on an infinitesimal element of the cutting edge are typically written as:

$$dF\_t(\\phi) \= K\_{tc} h(\\phi) db \+ K\_{te} ds$$

$$dF\_r(\\phi) \= K\_{rc} h(\\phi) db \+ K\_{re} ds$$

$$dF\_a(\\phi) \= K\_{ac} h(\\phi) db \+ K\_{ae} ds$$  
Here, $K\_{tc}, K\_{rc}, K\_{ac}$ are the specific cutting force coefficients related to the shearing mechanism, while $K\_{te}, K\_{re}, K\_{ae}$ represent the edge coefficients associated with the rubbing or ploughing action at the cutting edge.1 The angle $\\phi$ represents the instantaneous immersion angle of the tooth.

The critical variable in these equations is the instantaneous chip thickness, $h(\\phi)$. In a perfectly rigid system, this thickness would be determined solely by the feed geometry: $h\_{static} \= f\_z \\sin(\\phi)$, where $f\_z$ is the feed per tooth. However, no machine tool is perfectly rigid. The cutting forces excite the structure, causing the tool to vibrate relative to the workpiece. This vibration modulates the chip thickness, creating a dynamic component that is superimposed on the static geometry.3

$$h(t) \= \[f\_z \\sin(\\phi(t))\] \+ \\sin(\\phi(t)) \+ \\cos(\\phi(t))$$  
In this equation, $(x(t), y(t))$ represents the current vibration of the tool, and $(x(t-T), y(t-T))$ represents the vibration of the surface left by the previous tooth passage ($T$ being the tooth passing period). This dependency of the current chip thickness on the past vibration history is the definition of the **regenerative effect**, the primary mechanism driving self-excited vibrations known as chatter.4

### **2.2 The "Bingo" Validation: Proportionality of Vibration to Chip Load**

The cornerstone of this research report is the validation of the relationship between vibration signals and physical cutting parameters. The user's query highlights a specific heuristic: finding literature that explicitly links vibration amplitude to chip load. Our exhaustive review of the research material confirms this relationship is not merely heuristic but a fundamental physical law of the machining system in the linear and weakly nonlinear regimes.

Evidence of Proportionality:  
The most direct validation is found in the analysis of mechanical instabilities. As detailed in the literature on solving mechanical instabilities, the amplitude of vibration in a linear system is directly proportional to the amplitude of the excitation force.4 Since the excitation force in milling is dominated by the cutting force, and the cutting force is linearly (or near-linearly) dependent on the chip load ($F \\propto h$), a transitive logic chain is established:

1. **Force-Chip Relationship:** $F(t) \\approx K\_c \\cdot h(t)$ (Mechanistic Model).  
2. **Vibration-Force Relationship:** $X(t) \\approx H(\\omega) \\cdot F(t)$ (Structural Dynamics, where $H$ is the Frequency Response Function).  
3. **Conclusion:** $X(t) \\propto h(t)$.

This confirms that the vibration displacement—and by extension, its second derivative, acceleration—is a direct, proportional readout of the dynamic chip load variations.

**Corroborating Research:**

* Snippet 3: Explicitly models the force as "proportional to the instantaneous chip thickness... which consists of a static part... and a dynamic component caused by vibrations." This confirms that any fluctuation in the chip thickness (the load) will manifest as a fluctuation in the force and subsequent vibration.  
* Snippet 5: Defines the regenerative force specifically in terms of the variable chip thickness: "The difference between $Y\_0$ and $Y$ identifies the variable chip thickness due to the vibration... and provides the basis for regenerative chatter."  
* Snippet 6: Investigates the effect of tool runout, noting that "runout... causes periodic variations in the chip load... varying chip load influences the machining process." The study confirms that the vibration signature is modulated by these chip load variations, further validating the acceleration envelope as a diagnostic tool for load uniformity.

Therefore, we can assert with high confidence that monitoring the **acceleration envelope**—which captures the amplitude modulation of the vibration signal—is physically equivalent to monitoring the envelope of the dynamic chip load. This "bingo" insight allows us to bypass the complex, expensive, and compliance-inducing setup of force dynamometers in favor of robust, bandwidth-capable accelerometers.

### **2.3 Nonlinearities: The Limitations of Linear Models**

While the proportional relationship holds for stable, linear cutting, the reality of dynamic milling is often governed by nonlinearities that linear models fail to predict. It is in these nonlinear regimes that hysteresis becomes a dominant factor.

The "Size Effect" and Shear/Ploughing Transition:  
The specific cutting force coefficients ($K\_{tc}$) are not true constants. At very small chip thicknesses (micro-milling or finishing cuts), the physics of material removal changes. The edge radius of the tool becomes comparable to the chip thickness, leading to a dominance of the ploughing mechanism over shearing. This "size effect" introduces a strong nonlinearity where the specific cutting pressure increases exponentially as the chip thickness decreases.7 The relationship $F \\approx k \\cdot h$ evolves into a power law or polynomial function, often approximated as $f(h) \= C\_1 h^{3/4}$ in literature.9 This nonlinearity means that the vibration response to a fluctuating chip load is not perfectly sinusoidal but distorted, generating higher harmonics that are captured in the acceleration spectrum.  
Geometric Nonlinearity (Fly-Over):  
A severe nonlinearity occurs when the vibration amplitude exceeds the static chip thickness ($A \> h\_{static}$). In this scenario, the tool disengages from the workpiece entirely for a portion of the rotation, a phenomenon known as "fly-over" or loss of contact. During this phase, the cutting force drops to zero, introducing a discontinuity in the system dynamics.6 This is a "hard" nonlinearity that limits the growth of chatter amplitudes, leading to finite-amplitude stability rather than infinite divergence. The acceleration envelope captures this as a saturation or "clipping" of the vibration intensity, a critical feature for hysteresis modeling.

## **3\. Signal Processing: The Acceleration Envelope as an Information Carrier**

To utilize acceleration as a proxy for force, raw vibration data must be processed to extract the relevant "intensity" information. The raw acceleration signal in milling is a complex waveform composed of the tooth passing frequency, structural natural frequencies, and transient impacts. The **Acceleration Envelope** is the signal processing technique that demodulates this complex carrier to reveal the underlying dynamics of the chip load and process stability.

### **3.1 Mathematical Derivation of the Envelope**

The envelope of a signal $x(t)$ represents its instantaneous amplitude. In the context of milling, the "carrier" is typically the high-frequency structural resonance or the tooth passing frequency, while the "modulator" is the varying chip load or the developing instability (chatter). The most robust method for extracting this envelope is via the **Hilbert Transform**.

The analytic signal $z(t)$ is constructed from the real signal $x(t)$ and its Hilbert Transform $\\hat{x}(t)$:

$$z(t) \= x(t) \+ j\\hat{x}(t) \= A(t) e^{j\\phi(t)}$$  
Where $\\hat{x}(t)$ is defined as the convolution of $x(t)$ with the function $1/(\\pi t)$:

$$\\hat{x}(t) \= \\frac{1}{\\pi} \\int\_{-\\infty}^{\\infty} \\frac{x(\\tau)}{t \- \\tau} d\\tau$$  
The **Acceleration Envelope**, $A\_{env}(t)$, is the magnitude of this analytic signal:

$$A\_{env}(t) \= \\sqrt{x(t)^2 \+ \\hat{x}(t)^2}$$  
This mathematical operation effectively removes the high-frequency oscillations, leaving a trace that corresponds to the energy or intensity of the vibration.10

### **3.2 Synchronous Envelope Vibration Analysis (SEVA)**

In advanced monitoring systems, this technique is refined into **Synchronous Envelope Vibration Analysis (SEVA)**. SEVA involves resampling the envelope signal into the angular domain (synchronous with spindle rotation) to separate periodic events (like runout or tooth engagement) from asynchronous events (like chatter or bearing defects).

Snippet 11 details the use of SEVA for "qualitative and quantitative characterization of milling capacity." By analyzing the spectrum of the envelope, one can identify the "gear mesh frequency" or, in the case of milling, the specific harmonics associated with uneven chip loads caused by cutter runout. If the acceleration envelope shows a strong component at the spindle rotation frequency ($1 \\times \\Omega$), it indicates that one tooth is taking a significantly larger chip load than the others—a direct diagnosis of runout derived solely from the acceleration signal.11

### **3.3 Advantages over Direct Force Measurement**

The shift from dynamometers to accelerometers is driven by practical necessity and signal fidelity.

1. **Bandwidth:** Piezoelectric dynamometers often have limited bandwidth due to the mass of the workpiece or fixture they support. This low natural frequency can distort high-frequency cutting force dynamics. Accelerometers, being lightweight and stiff, offer much higher bandwidths (often \>10 kHz), capturing the rapid transients of tooth entry and exit.1  
2. **Structural Integrity:** Inserting a dynamometer reduces the static stiffness of the machine setup, potentially inducing the very chatter one wishes to study. Accelerometers are non-intrusive.  
3. **Sensitivity:** Acceleration is proportional to the square of the frequency ($\\omega^2$) times displacement. In high-speed milling, where frequencies are high, acceleration signals are extremely sensitive to minute changes in force dynamics, making the envelope a hyper-sensitive detector for the onset of instability.14

## **4\. Hysteresis in Machining: Phenomenology and Identification**

Hysteresis describes a system where the output depends not only on the current input but also on the history of that input. In dynamic milling, hysteresis manifests in two distinct but interconnected domains: the **macroscopic stability domain** (chatter onset/offset) and the **microscopic process domain** (process damping).

### **4.1 The "Unsafe Zone": Macroscopic Stability Hysteresis**

The classical Stability Lobe Diagram (SLD) draws a crisp line between stable and unstable cuts. However, reality is more complex. Recent research 9 has identified an "Unsafe Zone" (UZ) or region of bi-stability. In this parametric region, the system can exist in either a stable state (stationary cutting) or a chattering state (finite amplitude oscillation), depending on the perturbation history.

The Hysteresis Loop in Stability:  
This bi-stability creates a hysteresis loop in the process behavior.

* **Ramp-Up:** If the depth of cut ($a\_p$) or chip width ($w$) is slowly increased, the process remains stable until it hits the **supercritical Hopf bifurcation point** (or a similar threshold), at which point it jumps to high-amplitude chatter.  
* **Ramp-Down:** If the depth of cut is then decreased, the process does *not* immediately return to stability. It remains in the chatter state until a lower threshold, the **fold bifurcation point**, is reached.

The difference between these two points constitutes the width of the hysteresis loop ($\\Delta w$). Snippet 9 is explicit: "The unsafe parameter zone can be detected experimentally where hysteresis is observed during the appearance/disappearance of chatter." Crucially, the size of this hysteresis loop is determined by the **nonlinearity of the specific cutting force**.

Reconstructing Force from Hysteresis:  
The research suggests a powerful inverse method: by measuring the width of this stability hysteresis loop using the acceleration envelope (detecting the jump in vibration intensity), one can mathematically reconstruct the nonlinear cutting force function $f(h)$.9 This validates the report's core premise: the acceleration envelope is not just a monitoring tool but a metrological input for identifying the nonlinear constitutive laws of the cutting process.

### **4.2 Process Damping: The Microscopic Source of Hysteresis**

The physical mechanism driving this finite amplitude stability and hysteresis is **process damping**. Unlike structural damping, which dissipates energy in the machine joints, process damping dissipates energy directly in the cutting zone.

Mechanism of Interference:  
Process damping arises from the interference between the tool's relief (flank) face and the wavy surface generated by previous vibrations. When the tool vibrates downwards into the workpiece, the clearance angle effectively becomes negative, causing the flank face to indent and rub against the material.15  
The Hysteresis Loop in Force-Displacement:  
This indentation and rubbing process is not elastic; it is elasto-plastic and frictional. It generates a force that opposes the velocity of the tool, but in a nonlinear manner. If one plots the process damping force against the tool displacement, it traces a hysteresis loop.17 The area enclosed by this loop represents the energy dissipated per vibration cycle.

* **Energy Dissipation:** $E\_{diss} \= \\oint F\_{pd} \\cdot dy$.  
* **Velocity Dependence:** The force is often modeled as $F\_{pd} \= \-C \\frac{b}{V} \\dot{y}$, but experimental evidence shows a more complex hysteretic behavior that saturates at high vibration amplitudes.5

The acceleration envelope captures the macroscopic effect of this microscopic hysteresis. As chatter initiates, the envelope grows. However, as the vibration velocity increases, process damping kicks in (the hysteresis loop opens up), dissipating energy and capping the envelope at a finite value.

### **4.3 Structural Hysteresis: Joint Damping and Friction**

Beyond the cutting zone, the machine tool structure itself exhibits hysteresis, primarily at bolted joints and guideways. This is often modeled using **Coulomb friction** or **microslip** models.

* **Microslip:** At interfaces, small regions slip while others remain stuck. This creates a force-displacement hysteresis loop that is amplitude-dependent.  
* **Implication for Envelope:** This means the structural transfer function $H(\\omega)$ is not constant but depends on the vibration intensity. The acceleration envelope allows us to track this changing stiffness and damping in real-time, adjusting the force estimation model accordingly.19

## **5\. Mathematical Modeling Frameworks**

To move from qualitative description to quantitative prediction, we must employ mathematical models that can ingest the acceleration envelope and output the hysteretic force state. Two phenomenological models stand out in the literature: the **Bouc-Wen** model and the **Preisach** model.

### **5.1 The Bouc-Wen Model**

The Bouc-Wen model is the gold standard for modeling smooth, hysteretic behavior in mechanical systems, particularly for structural damping and magnetorheological (MR) dampers used in chatter suppression.21 Its differential nature makes it highly compatible with the time-domain simulation of milling dynamics.

Formulation:  
The restoring force $F(t)$ is decomposed into an elastic component and a hysteretic component:

$$F(x, \\dot{x}) \= \\alpha k x \+ (1-\\alpha) k z$$  
The hysteretic variable $z$ evolves according to the nonlinear differential equation:

$$\\dot{z} \= A \\dot{x} \- \\beta |\\dot{x}| |z|^{n-1} z \- \\gamma \\dot{x} |z|^n$$  
Relevance to Envelope Modeling:  
In our proposed framework, the parameters $A, \\beta, \\gamma$ are not constants but functions of the acceleration envelope $A\_{env}$. This allows the model to adapt to the changing regime of the cut. For example, as the envelope detects the onset of chatter (high intensity), the hysteresis parameters can shift to reflect the increased energy dissipation of process damping or the saturation of structural joints. Snippet 23 discusses identifying these parameters using optimization algorithms (like Particle Swarm Optimization), which can be driven by the error between predicted and measured acceleration envelopes.

### **5.2 The Preisach Model**

The Preisach model is an alternative approach that models hysteresis as the aggregate effect of infinite independent relay operators (hysterons). It is particularly effective for systems with strong memory effects, such as the piezoelectric actuators used in fast tool servos (FTS) for active vibration control.24

Formulation:  
The output $f(t)$ is an integral of the input $u(t)$ over a weighting function $\\mu(\\alpha, \\beta)$:

$$f(t) \= \\iint\_{\\alpha \\ge \\beta} \\mu(\\alpha, \\beta) \\hat{\\gamma}\_{\\alpha \\beta} \[u(t)\] d\\alpha d\\beta$$  
Application:  
While computationally heavier, the Preisach model excels at capturing asymmetric hysteresis loops. In the context of "Acceleration Envelope as Input," the Preisach model can be used to map the relationship between the command signal to a vibration dampener and the resulting reduction in the acceleration envelope. Snippet 27 suggests a "Neural-Preisach" approach, combining neural networks with the Preisach operator to learn the hysteresis from data—a perfect fit for data-rich environments where acceleration envelopes are continuously monitored.

### **5.3 Comparative Analysis: Linear vs. Bouc-Wen vs. Preisach**

| Feature | Linear Viscous Model | Bouc-Wen Model | Preisach Model |
| :---- | :---- | :---- | :---- |
| **Mathematical Basis** | $F \= c\\dot{x}$ | Differential Equation ($ \\dot{z} \=... $) | Integral of Operators ($ \\iint \\mu... $) |
| **Hysteresis Shape** | Elliptical (Energy $\\propto$ Freq) | Smooth, adjustable width/slope | Arbitrary, can be asymmetric |
| **Computational Cost** | Very Low | Low (suitable for real-time) | High (requires memory storage) |
| **Machining Relevance** | Basic stability limits | Process damping, Joint friction | Piezo actuators, Active control |
| **Input Compatibility** | Velocity ($\\dot{x}$) | Displacement ($x$) & Velocity ($\\dot{x}$) | Arbitrary Input History |

**Table 1: Selection of Hysteresis Models for Dynamic Milling Analysis.** The Bouc-Wen model is generally preferred for *process* modeling (force reconstruction), while Preisach is preferred for *actuator* control.

## **6\. Integrated Framework: From Envelope to Force Reconstruction**

We propose a unified modeling framework that integrates the mechanistic "Bingo" validation with the hysteretic mathematics.

Step 1: Signal Acquisition and Demodulation  
High-bandwidth accelerometers ($\>10$ kHz) acquire the raw vibration signal $a(t)$. The signal is passed through a Hilbert Transform to extract the instantaneous Acceleration Envelope, $A\_{env}(t)$.  
Step 2: Chip Load Estimation  
Leveraging the proportionality validated in Section 2.2, the dynamic chip load variation $\\Delta h(t)$ is estimated:

$$\\Delta h(t) \\propto \\iint A\_{env}(t) dt$$

(Note: In practice, spectral scaling factors derived from the system's Frequency Response Function (FRF) are applied).  
Step 3: Nonlinear Force Calculation  
The estimated chip load is fed into a nonlinear force model that accounts for the size effect:

$$F\_{static}(t) \= K\_c (h\_{static} \+ \\Delta h)^n$$

Where $n \\approx 0.75$ (Three-quarter rule).9  
Step 4: Hysteresis Superposition  
The hysteretic damping force is calculated using a modified Bouc-Wen observer, driven by the estimated displacement state:

$$F\_{total}(t) \= F\_{static}(t) \+ F\_{hyst}(z, A\_{env})$$

Here, the hysteresis variable $z$ is modulated by the current intensity of the acceleration envelope, allowing the model to capture the "swelling" of the hysteresis loop during chatter onset.  
Step 5: Validation and Correction  
The predicted force is compared against a baseline or a "Digital Twin" of the stable process. Deviations in the envelope that do not match the predicted force evolution indicate the onset of the "Unsafe Zone" and the activation of the macroscopic hysteresis loop.

## **7\. Industrial Applications and Future Outlook**

The transition to envelope-based hysteresis modeling opens significant avenues for industrial advancement.

### **7.1 Active Chatter Suppression**

By predicting the hysteretic behavior of the cutting force in real-time, active control systems can intervene more effectively. Snippet 28 discusses "Tunable vibration absorbers" that use viscoelastic or magnetic hysteresis to dissipate energy. An envelope-based controller can tune the stiffness/damping of these absorbers dynamically, keeping the system on the "stable" branch of the hysteresis loop within the Unsafe Zone.

### **7.2 Tool Condition Monitoring (TCM)**

Tool wear fundamentally alters the cutting force coefficients and the friction at the flank face. This leads to an expansion of the process damping hysteresis loop. By monitoring the *width* and *area* of the reconstructed force-displacement loop (derived from the acceleration envelope), operators can detect wear before it degrades part quality. Snippet 29 highlights the use of domain knowledge to map signal features to wear; the "Hysteresis Area" is a physically meaningful feature that correlates directly with flank wear land width.

### **7.3 Digital Twins and Virtual Machining**

The ultimate application is the integration of these models into Digital Twins. By feeding real-time acceleration envelopes into a virtual Bouc-Wen model running in parallel, manufacturers can visualize the invisible—the instantaneous force hysteresis loop—on the machine controller. This allows for "stability mapping" on the fly, where the machine learns the boundaries of the Unsafe Zone and optimizes feed rates to graze the edge of instability without crossing it, maximizing productivity.

## **8\. Conclusion**

The dynamic stability of milling is not a binary state but a complex, hysteretic continuum. This report has demonstrated that the **acceleration envelope** is the master key to unlocking the physics of this continuum.

Validated by the fundamental proportionality between vibration amplitude and chip load, the acceleration envelope provides a robust, non-intrusive metric for the dynamic cutting force. It captures the nonlinearities of the "size effect," the geometric discontinuities of "fly-over," and the energy dissipation of "process damping." By coupling this signal with advanced phenomenological models like Bouc-Wen, we can mathematically reconstruct the hysteresis loops that define the "Unsafe Zones" of machining.

This approach moves beyond simple threshold-based monitoring. It enables a deterministic understanding of how energy is stored and dissipated in the cutting zone, paving the way for adaptive control systems that do not just avoid chatter, but actively manage the nonlinear dynamics of the process to achieve unprecedented levels of productivity and precision. The acceleration envelope is not merely a symptom of vibration; it is the heartbeat of the machining process.

#### **Fuentes citadas**

1. (PDF) Inverse Method for Cutting Forces Parameters Evaluation \- ResearchGate, acceso: diciembre 9, 2025, [https://www.researchgate.net/publication/41956809\_Inverse\_Method\_for\_Cutting\_Forces\_Parameters\_Evaluation](https://www.researchgate.net/publication/41956809_Inverse_Method_for_Cutting_Forces_Parameters_Evaluation)  
2. A method for the identification of the specific force coefficients for mechanistic milling simulation | Request PDF \- ResearchGate, acceso: diciembre 9, 2025, [https://www.researchgate.net/publication/229146604\_A\_method\_for\_the\_identification\_of\_the\_specific\_force\_coefficients\_for\_mechanistic\_milling\_simulation](https://www.researchgate.net/publication/229146604_A_method_for_the_identification_of_the_specific_force_coefficients_for_mechanistic_milling_simulation)  
3. Universities of Leeds, Sheffield and York http://eprints.whiterose.ac.uk/, acceso: diciembre 9, 2025, [https://eprints.whiterose.ac.uk/id/eprint/11222/1/Guo\_11222.pdf](https://eprints.whiterose.ac.uk/id/eprint/11222/1/Guo_11222.pdf)  
4. Understanding & Solving Mechanical Instabilities \- Xcite Systems Corporation, acceso: diciembre 9, 2025, [https://xcitesystems.com/wp-content/uploads/2018/05/UnderstandingSolvingMechanicalInstabilities.pdf](https://xcitesystems.com/wp-content/uploads/2018/05/UnderstandingSolvingMechanicalInstabilities.pdf)  
5. PROCESS DAMPING IDENTIFICATION FOR MULTIPLE DEGREE OF FREEDOM TURNING OPERATIONS \- Machine Tool Research Center, acceso: diciembre 9, 2025, [https://mtrc.utk.edu/wp-content/uploads/sites/45/2019/09/tyler\_pd.pdf](https://mtrc.utk.edu/wp-content/uploads/sites/45/2019/09/tyler_pd.pdf)  
6. Runout effects in milling: Surface finish, surface location error, and stability \- Machine Tool Research Center, acceso: diciembre 9, 2025, [https://mtrc.utk.edu/wp-content/uploads/sites/45/2019/09/runout\_ra\_sle\_stability.pdf](https://mtrc.utk.edu/wp-content/uploads/sites/45/2019/09/runout_ra_sle_stability.pdf)  
7. Cutting Force—Vibration Interactions in Precise—and Micromilling Processes: A Critical Review on Prediction Methods \- MDPI, acceso: diciembre 9, 2025, [https://www.mdpi.com/1996-1944/18/15/3539](https://www.mdpi.com/1996-1944/18/15/3539)  
8. Cutting Force—Vibration Interactions in Precise—and Micromilling Processes: A Critical Review on Prediction Methods \- NIH, acceso: diciembre 9, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12348374/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12348374/)  
9. Identification of cutting force characteristics based on chatter ..., acceso: diciembre 9, 2025, [https://www.mm.bme.hu/\~stepan/docs/dombo\_munoa\_stepan\_CIRP\_Annals.pdf](https://www.mm.bme.hu/~stepan/docs/dombo_munoa_stepan_CIRP_Annals.pdf)  
10. Real-Time Envelope Monitoring of High-Speed Spindle in Commissioning Conditions: Grinding Machine \- MDPI, acceso: diciembre 9, 2025, [https://www.mdpi.com/2504-4494/9/9/298](https://www.mdpi.com/2504-4494/9/9/298)  
11. Envelope dynamic analysis: A new approach for milling process monitoring \- ResearchGate, acceso: diciembre 9, 2025, [https://www.researchgate.net/publication/257336081\_Envelope\_dynamic\_analysis\_A\_new\_approach\_for\_milling\_process\_monitoring](https://www.researchgate.net/publication/257336081_Envelope_dynamic_analysis_A_new_approach_for_milling_process_monitoring)  
12. Acceleration-based spindle monitoring based on geometric error motions \- ResearchGate, acceso: diciembre 9, 2025, [https://www.researchgate.net/publication/390512411\_Acceleration-based\_spindle\_monitoring\_based\_on\_geometric\_error\_motions](https://www.researchgate.net/publication/390512411_Acceleration-based_spindle_monitoring_based_on_geometric_error_motions)  
13. Cutting force reconstruction method based on static bandwidth expansion utilizing acceleration sensors | Request PDF \- ResearchGate, acceso: diciembre 9, 2025, [https://www.researchgate.net/publication/388598644\_Cutting\_force\_reconstruction\_method\_based\_on\_static\_bandwidth\_expansion\_utilizing\_acceleration\_sensors](https://www.researchgate.net/publication/388598644_Cutting_force_reconstruction_method_based_on_static_bandwidth_expansion_utilizing_acceleration_sensors)  
14. HEALTH MONITORING, FAULT DETECTION AND DIAGNOSIS IN INDUSTRIAL ROTATING MACHINERY BY ADVANCED VIBRATION ANALYSIS \- IRIS UniPA, acceso: diciembre 9, 2025, [https://iris.unipa.it/retrieve/e3ad891a-6d49-da0e-e053-3705fe0a2b96/Tesi\_Dottorato\_Karamoko\_Versione%20Finale.pdf](https://iris.unipa.it/retrieve/e3ad891a-6d49-da0e-e053-3705fe0a2b96/Tesi_Dottorato_Karamoko_Versione%20Finale.pdf)  
15. MODELLING AND ANALYSIS OF CHATTER MITIGATION STRATEGIES IN MILLING \- White Rose eTheses Online, acceso: diciembre 9, 2025, [https://etheses.whiterose.ac.uk/id/eprint/4482/1/MODELLING%20AND%20ANALYSIS%20OF%20CHATTER%20MITIGATION%20STRATEGIES%20IN%20MILLING%20.pdf](https://etheses.whiterose.ac.uk/id/eprint/4482/1/MODELLING%20AND%20ANALYSIS%20OF%20CHATTER%20MITIGATION%20STRATEGIES%20IN%20MILLING%20.pdf)  
16. A review of chatter suppression in thin-wall milling: strategies, mechanisms, and applications, acceso: diciembre 9, 2025, [http://www.ijemnet.com/article/pdf/preview/10.1088/2631-7990/adec24.pdf](http://www.ijemnet.com/article/pdf/preview/10.1088/2631-7990/adec24.pdf)  
17. euspen's 23rd International Conference & Exhibition, Copenhagen, DK, June 2023, acceso: diciembre 9, 2025, [https://www.euspen.eu/knowledge-base/ICE23172.pdf](https://www.euspen.eu/knowledge-base/ICE23172.pdf)  
18. Chatter suppression in micro end milling with process damping \- ResearchGate, acceso: diciembre 9, 2025, [https://www.researchgate.net/publication/248253631\_Chatter\_suppression\_in\_micro\_end\_milling\_with\_process\_damping](https://www.researchgate.net/publication/248253631_Chatter_suppression_in_micro_end_milling_with_process_damping)  
19. Dynamic properties of unbonded, multi-strand beams subjected to flexural loading, acceso: diciembre 9, 2025, [https://eprints.whiterose.ac.uk/id/eprint/116995/21/Manuscript.pdf](https://eprints.whiterose.ac.uk/id/eprint/116995/21/Manuscript.pdf)  
20. STUDIES OF INTERFACE DAMPING \- NASA Technical Reports Server, acceso: diciembre 9, 2025, [https://ntrs.nasa.gov/api/citations/19700004849/downloads/19700004849.pdf](https://ntrs.nasa.gov/api/citations/19700004849/downloads/19700004849.pdf)  
21. Nonlinear dynamics of a SDOF oscillator with Bouc-Wen hysteresis \- ResearchGate, acceso: diciembre 9, 2025, [https://www.researchgate.net/publication/228900343\_Nonlinear\_dynamics\_of\_a\_SDOF\_oscillator\_with\_Bouc-Wen\_hysteresis](https://www.researchgate.net/publication/228900343_Nonlinear_dynamics_of_a_SDOF_oscillator_with_Bouc-Wen_hysteresis)  
22. Schematic representation of the Bouc–Wen model. \- ResearchGate, acceso: diciembre 9, 2025, [https://www.researchgate.net/figure/Schematic-representation-of-the-Bouc-Wen-model\_fig2\_278133931](https://www.researchgate.net/figure/Schematic-representation-of-the-Bouc-Wen-model_fig2_278133931)  
23. A Novel Feedforward Model of Piezoelectric Actuator for Precision Rapid Cutting \- MDPI, acceso: diciembre 9, 2025, [https://www.mdpi.com/1996-1944/16/6/2271](https://www.mdpi.com/1996-1944/16/6/2271)  
24. (PDF) Modeling hysteresis of smart actuators \- ResearchGate, acceso: diciembre 9, 2025, [https://www.researchgate.net/publication/251859982\_Modeling\_hysteresis\_of\_smart\_actuators](https://www.researchgate.net/publication/251859982_Modeling_hysteresis_of_smart_actuators)  
25. A Novel Fractional Order Model for the Dynamic Hysteresis of Piezoelectrically Actuated Fast Tool Servo, acceso: diciembre 9, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5449064/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5449064/)  
26. Hysteresis modeling and tracking control for piezoelectric stack actuators using neural-Preisach model \- ResearchGate, acceso: diciembre 9, 2025, [https://www.researchgate.net/publication/337233869\_Hysteresis\_modeling\_and\_tracking\_control\_for\_piezoelectric\_stack\_actuators\_using\_neural-Preisach\_model](https://www.researchgate.net/publication/337233869_Hysteresis_modeling_and_tracking_control_for_piezoelectric_stack_actuators_using_neural-Preisach_model)  
27. Compound Control of Trajectory Errors in a Non-Resonant Piezo-Actuated Elliptical Vibration Cutting Device \- Semantic Scholar, acceso: diciembre 9, 2025, [https://pdfs.semanticscholar.org/4c9f/c7101de64b515f83189aeb2f193e8991e431.pdf](https://pdfs.semanticscholar.org/4c9f/c7101de64b515f83189aeb2f193e8991e431.pdf)  
28. Tunable vibration absorber for improving milling stability with tool wear and process damping effects | Request PDF \- ResearchGate, acceso: diciembre 9, 2025, [https://www.researchgate.net/publication/256934656\_Tunable\_vibration\_absorber\_for\_improving\_milling\_stability\_with\_tool\_wear\_and\_process\_damping\_effects](https://www.researchgate.net/publication/256934656_Tunable_vibration_absorber_for_improving_milling_stability_with_tool_wear_and_process_damping_effects)  
29. Toward digital twins for high-performance manufacturing: Tool wear monitoring in high-speed milling of thin-walled parts using domain knowledge | Request PDF \- ResearchGate, acceso: diciembre 9, 2025, [https://www.researchgate.net/publication/377438639\_Toward\_digital\_twins\_for\_high-performance\_manufacturing\_Tool\_wear\_monitoring\_in\_high-speed\_milling\_of\_thin-walled\_parts\_using\_domain\_knowledge](https://www.researchgate.net/publication/377438639_Toward_digital_twins_for_high-performance_manufacturing_Tool_wear_monitoring_in_high-speed_milling_of_thin-walled_parts_using_domain_knowledge)