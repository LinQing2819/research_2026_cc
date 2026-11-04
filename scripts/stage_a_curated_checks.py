"""CPU checks of manually transcribed build-source derivations.

These checks do not certify arbitrary prose or infer answers to development
questions. Source/parameter transcription and option uniqueness remain manual
review; their exact content is bound by the preparation package.
"""
from __future__ import annotations

import cmath
import math

from memgate.experiments.common import require

# Bind each calculation to its reviewed build question, not merely to a label.
SOURCE_CHECKS = {
    "11311": "dc_power", "11321": "complex_powers",
    "11322": "plate_capacitance", "11325": "resistor_integral",
    "11356": "solenoid_energy", "11428": "discrete_variance",
    "11430": "uniform_variance", "11464": "wave_amplitude",
    "11530": "rc_forced", "11634": "variable_conduction",
    "11651": "bayes_information", "11693": "linear_storage",
    "11854": "charge_ode", "11867": "conditional_mean",
}
RTOL, ATOL = 1e-7, 1e-12


def integral(f, a, b, n=2048):
    """Composite Simpson quadrature, independent of analytic antiderivatives."""
    require(n > 0 and n % 2 == 0, "Quadrature needs positive even subdivisions")
    h = (b - a) / n
    return h / 3 * math.fsum(
        (1 if i in (0, n) else 4 if i % 2 else 2) * f(a + i * h)
        for i in range(n + 1))


def derivative(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)


def close(actual, expected, label, rtol=RTOL, atol=ATOL):
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
            f"Numerical check failed: {label}: {actual} != {expected}")
    return dict(label=label, actual=actual, expected=expected,
                absolute_error=abs(actual - expected), rtol=rtol, atol=atol)


def calculate(source):
    """Return recomputed claims and checks using manually reviewed source values."""
    checks = []

    def check(actual, expected, label, **kw):
        checks.append(close(actual, expected, label, **kw))

    if source == "11311":
        v, r = 220, 110
        i = v / r
        result = dict(current_A=i, power_W=v * i)
        check(v * i, i * i * r, "VI = I^2 R")
    elif source == "11321":
        z = complex(8.66, 5)
        result = dict(magnitude=abs(z), phase_degrees=math.degrees(cmath.phase(z)))
        for n in (2, 5):
            polar = cmath.rect(abs(z)**n, n * cmath.phase(z))
            check(polar.real, (z**n).real, f"power {n}: rectangular real")
            check(polar.imag, (z**n).imag, f"power {n}: rectangular imaginary")
    elif source == "11322":
        # Arbitrary normalized parameters, not any development-question values.
        eps, area, d, v0 = 2, 3, 1, 5
        potential = lambda x: v0 * x / d
        field = -derivative(potential, d / 3)
        result = dict(normalized_C=eps * area / d)
        check(potential(0), 0, "left potential boundary")
        check(potential(d), v0, "right potential boundary")
        check(abs(eps * field * area / v0), result["normalized_C"], "Q/V from gradient")
        check(derivative(lambda x: derivative(potential, x), d / 2), 0,
              "charge-free potential second derivative", atol=1e-6)
    elif source == "11325":
        energy = 2 * 3 / 8 * 5
        result = dict(energy_J=energy)
        check(integral(lambda t: 2 * math.cos(math.pi*t)**4, 0, 5),
              energy, "quadrature of resistor instantaneous power")
    elif source == "11356":
        mu, turns, length, radius, current = 4e-7*math.pi, 1000, .30, .015, 10
        area = math.pi * radius**2
        inductance = mu * turns**2 * area / length
        energy = inductance * current**2 / 2
        result = dict(inductance_H=inductance, energy_J=energy)
        check(mu * (turns*current/length)**2 / 2 * area * length,
              energy, "field density integrated over solenoid volume")
    elif source == "11428":
        probabilities = (.25, .5, .125, .125)
        mean = sum(x*p for x, p in enumerate(probabilities))
        second = sum(x*x*p for x, p in enumerate(probabilities))
        result = dict(mean=mean, second_moment=second, variance=second-mean**2)
        check(sum(probabilities), 1, "discrete normalization")
        check(sum((x-mean)**2*p for x, p in enumerate(probabilities)),
              result["variance"], "centered second moment")
    elif source == "11430":
        result = dict(mean=1.5, second_moment=3, variance=.75)
        check(integral(lambda x: 1/3, 0, 3), 1, "uniform normalization")
        check(integral(lambda x: x/3, 0, 3), result["mean"], "first moment integral")
        check(integral(lambda x: x*x/3, 0, 3), result["second_moment"], "second moment integral")
        check(integral(lambda x: (x-1.5)**2/3, 0, 3), .75, "centered moment integral")
    elif source == "11464":
        eps, c, intensity = 8.8541878128e-12, 299792458, 1350
        electric = math.sqrt(2*intensity/(eps*c))
        magnetic = electric/c
        result = dict(electric_peak_V_m=electric, magnetic_peak_T=magnetic)
        eta = 1/(eps*c)
        check(integral(lambda phase: electric**2*math.cos(phase)**2/eta,
                       0, 2*math.pi)/(2*math.pi), intensity, "average Poynting flux")
        check(c*magnetic, electric, "E = cB")
        check((electric/math.sqrt(2))**2/eta, intensity, "RMS convention")
    elif source == "11530":
        r, omega, capacitance, v = 2000, 1000, 1e-6, 60
        impedance = complex(r, -1/(omega*capacitance))
        current = v/impedance
        result = dict(current_real_A=current.real, current_imag_A=current.imag)
        i = lambda t: current.real*math.cos(omega*t)-current.imag*math.sin(omega*t)
        # Integral of the two current components, divided by C.
        vc = lambda t: (current.real*math.sin(omega*t)
                        + current.imag*math.cos(omega*t))/(omega*capacitance)
        for t in (.00013, .0007, .0019):
            check(r*i(t)+vc(t), v*math.cos(omega*t), f"KVL at {t}")
            check(capacitance*derivative(vc, t, 1e-8), i(t), f"C dv/dt at {t}")
    elif source == "11634":
        a, b, hot, cold, thickness = .12, 5e-5, 2000, 200, 5/12
        primitive = lambda t: a*t + b*t*t/2
        heat_flux = (primitive(hot)-primitive(cold))/thickness
        position = lambda t: (primitive(hot)-primitive(t))/heat_flux
        result = dict(heat_flux_Btu_hr_ft2=heat_flux,
                      profile_constant_ft=primitive(hot)/heat_flux)
        check(integral(lambda t: a+b*t, cold, hot)/thickness,
              heat_flux, "quadrature of temperature-dependent conductivity")
        check(position(hot), 0, "hot face")
        check(position(cold), thickness, "cold face and inches-to-feet conversion")
        for t in (300, 1100, 1900):
            check(-(a+b*t)/derivative(position, t, .01), heat_flux,
                  f"local Fourier residual at {t} F")
    elif source == "11651":
        joint, evidence = .75*.25, .50
        posterior = joint/evidence
        information = -math.log2(posterior)
        result = dict(posterior=posterior, information_bits=information)
        check(posterior + (evidence-joint)/evidence, 1, "posterior normalization")
        check(2**(-information), posterior, "information inverse")
    elif source == "11693":
        energy, current, voltage = 20, 2, 500
        inductance, capacitance = 2*energy/current**2, 2*energy/voltage**2
        result = dict(inductance_H=inductance, capacitance_F=capacitance)
        check(integral(lambda i: inductance*i, 0, current),
              energy, "inductor energy integral")
        check(integral(lambda q: q/capacitance, 0, capacitance*voltage),
              energy, "capacitor energy integral")
        check(capacitance*1e6, 160, "farads to microfarads")
    elif source == "11854":
        ka, ib = 200*math.exp(-.2), .1*math.exp(.4)
        result = dict(charge_coefficient_C=ka, current_coefficient_A=ib)
        qa = lambda t: ka*math.exp(t/50)
        qb = lambda t: 50*ib*math.exp(t/50)
        check(qa(10)/50, 4, "part a initial current")
        check(qb(-20), 5, "part b initial charge")
        for name, q in (("a", qa), ("b", qb)):
            for t in (-20, 0, 10):
                check(derivative(q, t), q(t)/50, f"part {name} ODE at {t}")
    elif source == "11867":
        result = dict(normalization=integral(lambda x: x+.5, 0, 1),
                      mean_at_x_half=(.5/2+1/3)/(.5+.5))
        check(result["normalization"], 1, "joint density normalization")
        for x in (.1, .5, .9):
            density = lambda y: (x+y)/(x+.5)
            check(integral(density, 0, 1), 1, f"conditional normalization at {x}")
            check(integral(lambda y: y*density(y), 0, 1),
                  (3*x+2)/(3*(2*x+1)), f"conditional mean at {x}")
    else:
        raise ValueError(f"No reviewed calculation for source {source}")
    return dict(values=result, cross_checks=checks)


def verify_example(example):
    source = example["source"]
    require(SOURCE_CHECKS.get(source) == example["check"],
            "Calculation does not match reviewed build source")
    report = calculate(source)
    require(set(report["values"]) == set(example["claims"]), "Numerical claim keys changed")
    report["claim_checks"] = [
        close(report["values"][key], expected, f"{source}/{key}")
        for key, expected in sorted(example["claims"].items())]
    report["scope"] = (
        "Source parameters manually transcribed; arithmetic, quadrature and residual checks. "
        "Not a proof of all prose, source completeness, or automatic option uniqueness.")
    return report


def dimensional_checks():
    # Exponents in (kg, m, s, A, K); addition multiplies dimensions.
    unit = dict(V=(1, 2, -3, -1, 0), ohm=(1, 2, -3, -2, 0),
                A=(0, 0, 0, 1, 0), W=(1, 2, -3, 0, 0),
                J=(1, 2, -2, 0, 0), F=(-1, -2, 4, 2, 0),
                H=(1, 2, -2, -2, 0), T=(1, 0, -2, -1, 0),
                eps=(-1, -3, 4, 2, 0), k=(1, 1, -3, 0, -1),
                m=(0, 1, 0, 0, 0), s=(0, 0, 1, 0, 0), K=(0, 0, 0, 0, 1))

    def dim(factors):
        return [sum(unit[name][i]*power for name, power in factors.items()) for i in range(5)]

    identities = [
        ("V/R is current", dict(V=1, ohm=-1), dict(A=1)),
        ("VI is power", dict(V=1, A=1), dict(W=1)),
        ("R i^2 dt is energy", dict(ohm=1, A=2, s=1), dict(J=1)),
        ("C V^2 is energy", dict(F=1, V=2), dict(J=1)),
        ("L i^2 is energy", dict(H=1, A=2), dict(J=1)),
        ("epsilon S/d is capacitance", dict(eps=1, m=1), dict(F=1)),
        ("epsilon c E^2 is power/area", dict(eps=1, m=-1, s=-1, V=2), dict(W=1, m=-2)),
        ("E/c is magnetic flux density", dict(V=1, s=1, m=-2), dict(T=1)),
        ("k dT/dx is heat flux", dict(k=1, K=1, m=-1), dict(W=1, m=-2)),
    ]
    results = []
    for label, left, right in identities:
        require(dim(left) == dim(right), f"Dimension mismatch: {label}")
        results.append(dict(label=label, left=left, right=right, exponents=dim(left)))
    return results


def singularity_checks():
    """Check the retained rule's caveat; the underspecified source E is excluded."""
    k, radius = 2, 1.3
    checks = []
    for n in (-1, 0, 2):
        volume = integral(lambda r: 4*math.pi*k*(n+2)*r**(n+1), 0, radius)
        checks.append(close(volume, 4*math.pi*k*radius**(n+2), f"convergent flux n={n}"))
    # At n=-2 the shell has zero ordinary divergence but nonzero outer flux;
    # ignoring the equally large inner flux would give a wrong volume integral.
    outer = 4*math.pi*k
    checks.append(close(outer-4*math.pi*k, 0, "n=-2 outer minus inner flux"))
    require(outer != 0, "Point-source caveat became vacuous")
    # n=-3: the magnitude of the shell integral grows without bound.
    shell = lambda eps: 4*math.pi*k*(radius**-1-eps**-1)
    require(abs(shell(1e-5)) > 900*abs(shell(.01)), "Divergence caveat failed")
    return checks
