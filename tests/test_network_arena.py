import math

import pytest

import fpm_cpp as fpm


def test_network_arena_is_soa_aligned():
    net = fpm.FpmNetwork(4, 2)

    assert net.n_daemons == 4
    assert net.n_pairs == 2
    assert net.arena_bytes > 0
    assert net.arena_aligned is True


def test_routing_tensor_is_flattened_nine_channel_copy_out():
    net = fpm.FpmNetwork(2)
    r9 = [float(i) for i in range(9)]

    net.set_routing_tensor(0, r9)
    out = net.routing_tensor(0)

    assert out == r9
    out[0] = 999.0
    assert net.routing_tensor(0)[0] == 0.0


def test_viscosity_update_matches_simulator_s9_k1_mobility_formula():
    net = fpm.FpmNetwork(1)
    r9 = [
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ]
    net.set_routing_tensor(0, r9)
    before = net.energy(0)

    out = net.viscosity_update_from_routing(0)

    s9 = math.sqrt(sum(x * x for x in r9) / 9.0)
    k1 = abs(r9[0] + r9[4] + r9[8])
    c_n = 1.0 / ((1.0 + k1) ** 0.2 * (1.0 + s9) ** 1.8)

    assert out.S9 == pytest.approx(s9)
    assert out.K1 == pytest.approx(k1)
    assert out.C_N == pytest.approx(c_n)
    assert out.kappa == pytest.approx(c_n)
    assert out.Omega == pytest.approx(max(0.50, min(0.85, 1.0 - c_n)))
    assert net.energy(0) < before
    assert net.total_route_spend > 0.0


def test_resolve_torsion_link_requires_zombie_gate():
    net = fpm.FpmNetwork(2)

    with pytest.raises(RuntimeError, match="ZOMBIE"):
        net.resolve_torsion_link(0, 1)


def test_resolve_torsion_link_pulls_partner_and_records_exhaust():
    net = fpm.FpmNetwork(2)
    net.set_energy(0, 0.1, 1.0)
    net.set_energy(1, 0.9, 1.0)
    net.set_mode(0, fpm.MODE_ZOMBIE)

    out = net.resolve_torsion_link(0, 1)

    assert out.joint_quantization_executed is True
    assert out.pull_exhaust == pytest.approx(0.7)
    assert out.tv_distance >= 0.0
    assert net.energy(1) == pytest.approx(0.2)
    assert net.mode(1) == fpm.MODE_ZOMBIE
    assert net.total_pull_exhaust == pytest.approx(0.7)


def test_force_zombie_starvation_is_ledger_gated():
    net = fpm.FpmNetwork(1)
    net.set_energy(0, 0.8, 1.0)

    net.force_zombie_starvation(0)

    assert net.mode(0) == fpm.MODE_ZOMBIE
    assert net.energy(0) == pytest.approx(0.2)
    assert net.total_pull_exhaust == pytest.approx(0.6)
