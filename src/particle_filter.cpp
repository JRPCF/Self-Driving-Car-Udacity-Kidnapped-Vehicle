/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 *
 * I used this as reference when I was stuck and modified functions to approximate it
 * https://github.com/darienmt/CarND-Kidnapped-Vehicle-P3
 */
#include "particle_filter.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "helper_functions.h"
using namespace std;
using std::string;
using std::vector;
using std::normal_distribution;
using std::numeric_limits;
void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  std::default_random_engine gen;
  
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
  return;
}
void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   std::default_random_engine gen;
   for(int i = 0; i < num_particles; i++){
     double x;
     double y;
     double theta;
     if (fabs(yaw_rate) < 0.00001) {  
      x = velocity * delta_t * cos(particles[i].theta);
      y = velocity * delta_t * sin(particles[i].theta);
      theta = 0;
     } 
     else {
      x = velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      y = velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      theta = yaw_rate * delta_t;
     }
     normal_distribution<double> dist_x(x, std_pos[0]);
     normal_distribution<double> dist_y(y, std_pos[1]);
     normal_distribution<double> dist_theta(theta, std_pos[2]);
     particles[i].x += dist_x(gen);
     particles[i].y += dist_y(gen);
     particles[i].theta += dist_theta(gen);
   }
}
void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for (unsigned int i = 0; i < observations.size(); i++) {
    double min = numeric_limits<double>::max();
    int id = -1;
    for (unsigned int j = 0; j < predicted.size(); j++) {
      double cur = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (cur < min) {
        min = cur;
        id = predicted[j].id;
      }
    }
    observations[i].id = id;
  }
}
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
   for (int i = 0; i < num_particles; i++) {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;
    //get predictions in sensor range 
    vector<LandmarkObs> predictions;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      if (fabs(map_landmarks.landmark_list[j].x_f - x) <= sensor_range && fabs(map_landmarks.landmark_list[j].y_f - y) <= sensor_range) {
        predictions.push_back(LandmarkObs{ map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f });
      }
    }
     
    //change to coordinate frame
    vector<LandmarkObs> os;
    for (unsigned int j = 0; j < observations.size(); j++) {
      os.push_back(LandmarkObs{observations[j].id, x+cos(theta)*observations[j].x - sin(theta)*observations[j].y, y+cos(theta)*observations[j].y+sin(theta)*observations[j].x });
    }
    dataAssociation(predictions,os);
    
     particles[i].weight = 1.0;
     for(unsigned int j = 0; j < os.size(); j++){
         LandmarkObs temp;
         for(unsigned int k = 0; k < predictions.size(); k++){
           if(os[j].id==predictions[k].id)
               temp=predictions[k];
         }
         particles[i].weight *= ( 1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp( -( pow(temp.x-os[j].x,2)/(2*pow(std_landmark[0], 2)) + (pow(temp.y-os[j].y,2)/(2*pow(std_landmark[1], 2))) ) );
     }
   }
}
void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
   std::vector<Particle> particles_2;
   std::vector<double> weight;
   std::default_random_engine gen;
   for (int i = 0; i < num_particles; i++) {
    weight.push_back(particles[i].weight);
   }
   std::discrete_distribution<> dist(weight.begin(), weight.end());
   for(int i = 0; i < num_particles; i++) {
        particles_2.push_back(particles[dist(gen)]);
   }
   particles=particles_2;
}
void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}
string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;
  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}