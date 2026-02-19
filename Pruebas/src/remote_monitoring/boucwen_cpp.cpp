#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

class BoucWenProcessor {
public:
    BoucWenProcessor(double fs = 2500.0,
                     double mass = 0.107,
                     double stiffness = 1424.0,
                     double damping = 3.73,
                     double alpha = 0.811,
                     double a_bw = 1.0,
                     double b_bw = 0.5,
                     double c_bw = 0.5,
                     double n_bw = 1.0,
                     double k_env = 0.0669,
                     double b_env = 0.8837,
                     double cutting_threshold_g = 0.05)
        : fs_(fs), mass_(mass), stiffness_(stiffness), damping_(damping), alpha_(alpha),
          a_bw_(a_bw), b_bw_(b_bw), c_bw_(c_bw), n_bw_(n_bw),
          k_env_(k_env), b_env_(b_env), cutting_threshold_g_(cutting_threshold_g) {
        set_fs(fs_);
        z_ = 0.0;
        v_state_ = 0.0;
        x_state_ = 0.0;
        prev_accel_ = 0.0;
    }

    void set_fs(double fs) {
        if (!(fs > 0.0)) throw std::runtime_error("fs must be > 0");
        fs_ = fs;
        dt_ = 1.0 / fs_;
        buf_len_ = (int)std::llround(fs_ * 0.5);
        if (buf_len_ < 64) buf_len_ = 64;
        accel_buf_.assign(buf_len_, 0.0f);
        force_buf_.assign(buf_len_, 0.0f);
        write_pos_ = 0;
        filled_ = 0;
    }

    py::object process_block(py::array_t<float, py::array::c_style | py::array::forcecast> force,
                             py::array_t<float, py::array::c_style | py::array::forcecast> accel,
                             double fs) {
        if (fs > 0.0 && std::fabs(fs - fs_) > 1.0) set_fs(fs);

        auto f = force.unchecked<1>();
        auto a = accel.unchecked<1>();
        const int blk = (int)f.shape(0);
        if ((int)a.shape(0) != blk) throw std::runtime_error("force and accel must have same length");
        if (blk <= 0) return py::none();

        // ring buffer update
        for (int i = 0; i < blk; i++) {
            accel_buf_[write_pos_] = a(i);
            force_buf_[write_pos_] = f(i);
            write_pos_++;
            if (write_pos_ >= buf_len_) write_pos_ = 0;
            filled_ = std::min(buf_len_, filled_ + 1);
        }
        if (filled_ < 64) return py::none();

        // block stats (force/accel)
        double force_mean = 0.0, accel_mean = 0.0;
        double force_rms = 0.0, accel_rms = 0.0;
        double force_min = 1e30, force_max = -1e30;
        double accel_min = 1e30, accel_max = -1e30;
        for (int i = 0; i < blk; i++) {
            const double fv = (double)f(i);
            const double av = (double)a(i);
            force_mean += fv;
            accel_mean += av;
            force_rms += fv * fv;
            accel_rms += av * av;
            force_min = std::min(force_min, fv);
            force_max = std::max(force_max, fv);
            accel_min = std::min(accel_min, av);
            accel_max = std::max(accel_max, av);
        }
        force_mean /= (double)blk;
        accel_mean /= (double)blk;
        force_rms = std::sqrt(force_rms / (double)blk);
        accel_rms = std::sqrt(accel_rms / (double)blk);
        const double force_pp = force_max - force_min;
        const double accel_pp = accel_max - accel_min;

        // envelope approximation: |accel|
        double env_rms = 0.0;
        double F_est_mean = 0.0;
        double F_est_peak = -1e30;
        for (int i = 0; i < blk; i++) {
            const double env = std::fabs((double)a(i));
            env_rms += env * env;
            const double Fest = k_env_ * env + b_env_;
            F_est_mean += Fest;
            F_est_peak = std::max(F_est_peak, Fest);
        }
        env_rms = std::sqrt(env_rms / (double)blk);
        F_est_mean /= (double)blk;
        const double cutting_flag = (env_rms > cutting_threshold_g_) ? 1.0 : 0.0;

        // integrate accel -> vel -> disp (simple trapezoid, stateful)
        std::vector<double> v_blk(blk);
        std::vector<double> x_blk(blk);
        double v = v_state_;
        double x = x_state_;
        double a_prev = prev_accel_;
        for (int i = 0; i < blk; i++) {
            const double a_now = (double)a(i);
            v = v + 0.5 * (a_prev + a_now) * dt_;
            x = x + 0.5 * (v_state_ + v) * dt_;
            v_state_ = v;
            x_state_ = x;
            a_prev = a_now;
            v_blk[(size_t)i] = v;
            x_blk[(size_t)i] = x;
        }
        prev_accel_ = a_prev;

        // RK4 Bouc-Wen on z
        std::vector<double> z_blk(blk);
        double z = z_;
        for (int i = 0; i < blk; i++) {
            const double v_i = v_blk[(size_t)i];
            auto f_dz = [&](double z_in) {
                const double abs_z = std::fabs(z_in) + 1e-8;
                const double sign_vz = ((v_i * z_in) >= 0.0) ? 1.0 : -1.0;
                return v_i * (a_bw_ - std::pow(abs_z, n_bw_) * (b_bw_ * sign_vz + c_bw_));
            };
            const double k1 = f_dz(z);
            const double k2 = f_dz(z + 0.5 * dt_ * k1);
            const double k3 = f_dz(z + 0.5 * dt_ * k2);
            const double k4 = f_dz(z + dt_ * k3);
            z = z + (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            z = std::max(-1.0, std::min(1.0, z));
            z_blk[(size_t)i] = z;
        }
        z_ = z;

        // KAN-PINN Bouc-Wen force
        std::vector<double> F_bw(blk);
        for (int i = 0; i < blk; i++) {
            const double a_i = (double)a(i);
            const double v_i = v_blk[(size_t)i];
            const double x_i = x_blk[(size_t)i];
            const double E_i = std::fabs((double)a(i));
            const double z_i = z_blk[(size_t)i];
            F_bw[(size_t)i] = mass_ * a_i + stiffness_ * x_i + damping_ * v_i
                            + alpha_ * stiffness_ * E_i + (1.0 - alpha_) * stiffness_ * z_i;
        }

        // scale to measured force range
        double f_std = 0.0;
        for (int i = 0; i < blk; i++) {
            const double dv = (double)f(i) - force_mean;
            f_std += dv * dv;
        }
        f_std = std::sqrt(f_std / (double)blk) + 1e-10;

        double bw_mean_raw = 0.0;
        for (int i = 0; i < blk; i++) bw_mean_raw += F_bw[(size_t)i];
        bw_mean_raw /= (double)blk;

        double bw_std_raw = 0.0;
        for (int i = 0; i < blk; i++) {
            const double d = F_bw[(size_t)i] - bw_mean_raw;
            bw_std_raw += d * d;
        }
        bw_std_raw = std::sqrt(bw_std_raw / (double)blk) + 1e-10;

        std::vector<double> F_bw_scaled(blk);
        for (int i = 0; i < blk; i++) {
            F_bw_scaled[(size_t)i] = (F_bw[(size_t)i] - bw_mean_raw) / bw_std_raw * f_std + force_mean;
        }

        double F_bw_mean = 0.0;
        double F_bw_peak = -1e30;
        for (int i = 0; i < blk; i++) {
            F_bw_mean += F_bw_scaled[(size_t)i];
            F_bw_peak = std::max(F_bw_peak, F_bw_scaled[(size_t)i]);
        }
        F_bw_mean /= (double)blk;

        double ss_res = 0.0;
        double ss_tot = 0.0;
        for (int i = 0; i < blk; i++) {
            const double fv = (double)f(i);
            const double e = fv - F_bw_scaled[(size_t)i];
            ss_res += e * e;
            const double d = fv - force_mean;
            ss_tot += d * d;
        }
        ss_tot += 1e-10;
        double R2_block = 1.0 - ss_res / ss_tot;
        if (R2_block < -1.0) R2_block = -1.0;
        if (R2_block > 1.0) R2_block = 1.0;

        double z_mean = 0.0;
        double z_abs_max = 0.0;
        for (int i = 0; i < blk; i++) {
            z_mean += z_blk[(size_t)i];
            z_abs_max = std::max(z_abs_max, std::fabs(z_blk[(size_t)i]));
        }
        z_mean /= (double)blk;
        const double z_last = z_blk[(size_t)blk - 1];

        // hysteresis (very rough): area in (E, F_bw_scaled)
        double hyst_area = 0.0;
        for (int i = 0; i < blk - 1; i++) {
            const double E0 = std::fabs((double)a(i));
            const double E1 = std::fabs((double)a(i + 1));
            const double F0 = F_bw_scaled[(size_t)i];
            const double F1 = F_bw_scaled[(size_t)i + 1];
            hyst_area += E0 * F1 - E1 * F0;
        }
        hyst_area = std::fabs(hyst_area) * 0.5;
        double total_energy = 0.0;
        for (int i = 0; i < blk; i++) total_energy += std::fabs((double)f(i));
        total_energy += 1e-10;
        double hyst_energy_pct = hyst_area / total_energy * 100.0;
        if (hyst_energy_pct > 100.0) hyst_energy_pct = 100.0;

        double vel_rms = 0.0;
        double disp_rms = 0.0;
        for (int i = 0; i < blk; i++) {
            vel_rms += v_blk[(size_t)i] * v_blk[(size_t)i];
            disp_rms += x_blk[(size_t)i] * x_blk[(size_t)i];
        }
        vel_rms = std::sqrt(vel_rms / (double)blk);
        disp_rms = std::sqrt(disp_rms / (double)blk);

        // output 23 floats, matching BOUCWEN_RESULT_NAMES
        auto out = py::array_t<float>(23);
        auto o = out.mutable_unchecked<1>();
        o(0) = (float)F_est_mean;
        o(1) = (float)F_est_peak;
        o(2) = (float)env_rms;
        o(3) = 0.0f; // freq_dom_Hz
        o(4) = 0.0f; // freq_dom_mag
        o(5) = 0.0f; // THD_accel_pct
        o(6) = 0.0f; // THD_force_pct
        o(7) = (float)cutting_flag;
        o(8) = (float)force_rms;
        o(9) = (float)accel_rms;
        o(10) = (float)force_pp;
        o(11) = (float)accel_pp;
        o(12) = (float)F_bw_mean;
        o(13) = (float)F_bw_peak;
        o(14) = (float)z_mean;
        o(15) = (float)z_last;
        o(16) = (float)z_abs_max;
        o(17) = (float)alpha_;
        o(18) = (float)R2_block;
        o(19) = (float)hyst_area;
        o(20) = (float)hyst_energy_pct;
        o(21) = (float)vel_rms;
        o(22) = (float)disp_rms;
        return out;
    }

private:
    double fs_ = 2500.0;
    double dt_ = 1.0 / 2500.0;

    int buf_len_ = 1250;
    int write_pos_ = 0;
    int filled_ = 0;

    std::vector<float> accel_buf_;
    std::vector<float> force_buf_;

    double z_ = 0.0;
    double v_state_ = 0.0;
    double x_state_ = 0.0;
    double prev_accel_ = 0.0;

    double mass_ = 0.107;
    double stiffness_ = 1424.0;
    double damping_ = 3.73;
    double alpha_ = 0.811;

    double a_bw_ = 1.0;
    double b_bw_ = 0.5;
    double c_bw_ = 0.5;
    double n_bw_ = 1.0;

    double k_env_ = 0.0669;
    double b_env_ = 0.8837;
    double cutting_threshold_g_ = 0.05;
};

PYBIND11_MODULE(boucwen_cpp, m) {
    m.doc() = "Bouc-Wen + KAN-PINN inference engine (C++/pybind11)";

    py::class_<BoucWenProcessor>(m, "BoucWenProcessor")
        .def(py::init<double, double, double, double, double, double, double, double, double, double, double, double>(),
             py::arg("fs") = 2500.0,
             py::arg("mass") = 0.107,
             py::arg("stiffness") = 1424.0,
             py::arg("damping") = 3.73,
             py::arg("alpha") = 0.811,
             py::arg("a_bw") = 1.0,
             py::arg("b_bw") = 0.5,
             py::arg("c_bw") = 0.5,
             py::arg("n_bw") = 1.0,
             py::arg("k_env") = 0.0669,
             py::arg("b_env") = 0.8837,
             py::arg("cutting_threshold_g") = 0.05)
        .def("set_fs", &BoucWenProcessor::set_fs, py::arg("fs"))
        .def("process_block", &BoucWenProcessor::process_block,
             py::arg("force"), py::arg("accel"), py::arg("fs") = 0.0);
}
