#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>


// From https://cplusplus.com/forum/general/216928/
double interp1d( std::vector<double> &xData, std::vector<double> &yData, double x)
{
    int size = xData.size();

    int i = 0;                                                                  // find left end of interval for interpolation
    if ( x >= xData[size - 2] )                                                 // special case: beyond right end
    {
        i = size - 2;
    }
    else
    {
        while ( x > xData[i+1] ) i++;
    }
    double xL = xData[i], yL = yData[i], xR = xData[i+1], yR = yData[i+1];      // points on either side (unless beyond ends)

    // Not allowing extrapolation
    if ( x < xL ) yR = yL;
    if ( x > xR ) yL = yR;

    double dydx = ( yR - yL ) / ( xR - xL );                                    // gradient

    return yL + dydx * ( x - xL );                                              // linear interpolation
}


std::vector<std::vector<double>> loadTrajectory(const std::string& trajectoryFileName) {
    
    // Read in trajectory file
    std::vector<std::vector<double>> trajectory;
    std::ifstream file(trajectoryFileName);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::vector<double> row;
            std::stringstream ss(line);
            double value;
            while (ss >> value) {
                row.push_back(value);
                if (ss.peek() == ',') {
                    ss.ignore();
                }
            }
            trajectory.push_back(row);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << trajectoryFileName << std::endl;
    }

    return trajectory;
}


class Trajectory {
public:
    Trajectory(const std::string& trajectoryFileName) {
        // Read in the trajectory filename
        trajectory = loadTrajectory(trajectoryFileName);

        // Get the t, x, and y values
        for (int i=0; i < trajectory.size(); i++) {
            t.push_back(trajectory[i][0]);
            x.push_back(trajectory[i][1]);
            y.push_back(trajectory[i][2]);
        }

        // Get the velocity at each point along the trajectory
        double dt;
        for (int i=0; i < trajectory.size() - 1; i++) {
            dt = t[i+1] - t[i];
            x.push_back((x[i+1] - x[i]) / dt);
            y.push_back((y[i+1] - y[i]) / dt);
        }
        x.push_back(0.0);  // add (t_final, 0) to the end
        y.push_back(0.0);  // add (t_final, 0) to the end
    }

    std::vector<double> positionAtT(const double time) { return {interp1d(t, x, time), interp1d(t, y, time)}; }
    std::vector<double> velocityAtT(const double time) { return {interp1d(t, vx, time), interp1d(t, vy, time)}; }

    // Getters and setters
    std::vector<double> getT() const { return t; }

private:
    std::vector<std::vector<double>> trajectory;
    std::vector<double> t;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> vx;
    std::vector<double> vy;
};


class Agent {
public:
    Agent(double mass_kg, double thrustCapacity_N): mass(mass_kg), thrusterCapacity(thrustCapacity_N) {};


    // Execute one step
    void step(double dt) {

        // Get the force vector
        thrusterForce[0] = thrusterFiring * thrusterCapacity * sin(thrusterTheta);      // Fx = I(firing) * |F|sin(theta)
        thrusterForce[1] = thrusterFiring * thrusterCapacity * cos(thrusterTheta);      // Fx = I(firing) * |F|cos(theta)

        // Update the acceleration and velocity
        for (int i=0; i < acceleration.size(); i++) {
            acceleration[i] = thrusterForce[i] / mass;                                  // a = F/m
            position[i] += (velocity[i] * dt) + (0.5 * acceleration[i] * dt * dt);      // x = x0 + v0*dt + (1/2)*a*dt^2
            velocity[i] += acceleration[i] * dt;                                        // v = v0 + a*dt
        }
    }


    // Log current state
    void logState(const double currentTime, std::ofstream& logFile) {
        logFile << currentTime << ",";
        for (double r : position) {
            logFile << r << ",";
        }
        for (double v : velocity) {
            logFile << v << ",";
        }
        for (int i = 0; i < acceleration.size(); ++i) {
            logFile << acceleration[i];
            if (i < acceleration.size() - 1) {
                logFile << ",";
            }
        }
        logFile << "\n";
    }

    // Getters
    double getMass() const { return mass; }
    double getThrusterCapacity() const { return thrusterCapacity; }
    bool isThrusterFiring() const { return thrusterFiring; }
    double getThrusterTheta() const { return thrusterTheta; }
    std::vector<double> getThrusterForce() const { return thrusterForce; }
    std::vector<double> getPosition() const { return position; }
    std::vector<double> getVelocity() const { return velocity; }
    std::vector<double> getAcceleration() const { return acceleration; }

    // Setters
    void setMass(double m) { mass = m; }
    void setThrusterCapacity(double capacity) { thrusterCapacity = capacity; }
    void setThrusterFiring(bool firing) { thrusterFiring = firing; }
    void setThrusterTheta(double theta) { thrusterTheta = theta; }
    void setThrusterForce(const std::vector<double>& force) { thrusterForce = force; }
    void setPosition(const std::vector<double>& pos) { position = pos; }
    void setVelocity(const std::vector<double>& vel) { velocity = vel; }
    void setAcceleration(const std::vector<double>& acc) { acceleration = acc; }

private:
    double mass;                                        // [kg]     Mass of the agent
    double thrusterCapacity;                            // [N]      Magnitude of the force exerted by the thruster
    bool thrusterFiring;                                // [bool]   Whether the thruster is firing
    double thrusterTheta = 0.0;                         // [rad]    Angle between thruster and y-axis
    std::vector<double> thrusterForce = {0.0, 0.0};     // [N]      Force vector exerted by the thruster
    std::vector<double> position = {0.0, 0.0};          // [m]      Position of the agent (x, y)
    std::vector<double> velocity = {0.0, 0.0};          // [m/s]    Velocity of the agent (vx, vy)
    std::vector<double> acceleration = {0.0, 0.0};      // [m^2/s]  Acceleration of the agent (ax, ay)
};


class Controller {
public:
    Controller(const std::string& trajectoryFileName) : trajectory(trajectoryFileName) {
    }

    void policy(Agent& agent) {
        // HERE IS THE CRUX OF THE CONTROLLING POLICY
        agent.setVelocity({1.0, 0.0});
    }

    double get_t_final() const { return *(trajectory.getT().end() - 1); }

private:
    Trajectory trajectory;
};


class Sim{
public:
    Sim(Agent agent, const Controller& controller, const std::string& logFileName): agent(agent), controller(controller) {
        // Make a log file
        logFile.open(logFileName);
        t_final = controller.get_t_final();
    }

    // Run simulation!
    void run_sim(const double dt) {

        // Write log file header and initial log line
        logFile << "time,x,y,velocity_x,velocity_y,acceleration_x,acceleration_y\n";
        double time = 0;
        agent.logState(time, logFile);

        // Simulation loop
        while (time < t_final) {
            step(dt);
            time += dt;
            agent.logState(time, logFile);
        }
    };


private:
    Agent agent;
    Controller controller;
    std::vector<std::vector<double>> trajectory;
    std::ofstream logFile;
    double t_final;

    void step(double dt) {

        // Update the agent according to the policy and then take a step
        controller.policy(agent);
        agent.step(dt);
    }
};


int main() {

    // Define the trajectory file and log file names
    std::string trajectoryFile = "/Users/adamboesky/Research/RL_CMAES/trajectories/straight.csv";
    std::string logFileName = "/Users/adamboesky/Research/RL_CMAES/sim_results/test.csv";

    // Create an agent with initial mass and thrust capacity
    Agent agent(1000.0, 5000.0);

    // Create a controller
    Controller controller(trajectoryFile);

    // Create a simulation
    Sim simulation(agent, controller, logFileName);

    // Run the simulation with a time step of 0.1 seconds
    simulation.run_sim(0.1);

    return 0;
}
