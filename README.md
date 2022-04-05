# Earthquake_Prediction_Challenge

## Introduction

An earthquake early warning system limits the impact of future earthquakes on communities living nearby. The system detects an earthquake at its very beginning and rapidly issues alerts to users in its path. The alert outpaces strong earthquake shaking and may provide critical time to take basic protective actions such as seeking cover or exiting a building. If you would like to know more about EEW systems, you can refer to [this paper](https://www.researchgate.net/publication/330744338_Earthquake_Early_Warning_Advances_Scientific_Challenges_and_Societal_Needs).

!["Earthquake Early Warning System"](./fjg1.png)

*Figure 1 | The principle and main components of earthquake early warning system.*

The backbone of an EEW system is a network of seismic instruments continuously recording the ground motion (no. 1 in Figure 1) and transmitting data (no. 2) to a remote server (no. 3). When an earthquake occurs, it produces two kinds of seismic waves; the primary waves (P-waves), which travel fast and produce low-intensity shaking, and secondary waves (S-waves), which travel slower, but produce higher-intensity shaking that often causes structural damage. The EEW system detects earthquake primary waves and issues an alert for a region that is expected to experience intense shaking (no. 4). As the wireless communication of the EEW system is faster than the speed of seismic waves in the geological environment, the alert may outpace the shaking.

### Grillo/OpenEEW earthquake early warning system

Grillo, a social enterprise, developed a novel EEW solution that uses low-cost, high-performance seismic sensors and a cloud-based data platform. In 2020 Grillo's technology became an open-source solution, “OpenEEW”, supported by IBM and the Linux Foundation. OpenEEW now has a global community of developers contributing constantly to its development. It features the information for making the seismic sensors and software for earthquake detection and alerting. It also features over 1TB of sensor data that has been published since 2017 to advance research through the AWS Open Data program. OpenEEW community, with Grillo’s leadership and support, is now generating community-run networks in Nepal, New Zealand, and Chile. Since the official launch in August 2020 OpenEEW has attracted over 300 members to the community, and there are over 30 contributors to the technology in Github. Data is published through [AWS open data registry](https://registry.opendata.aws/grillo-openeew/).

### Seismic station and earthquake recordings

A seismic station is a device that records ground motion. In particular Grillo devices record the **acceleration** of the ground motion. The device uses a similar (albeit more precise) accelerometer sensor than you would find in consumer electronics such as smartphones. The device records the ground acceleration in three components - two horizontal ones and one vertical (x, y, z).

!["Untitled"](./fig2.png)

***Figure 2 |** Left: Description of Grillo the sensor. Right: A three-component recording of an earthquake arriving at a single seismic station.*

As mentioned above, once an earthquake occurs, it generates two kinds of waves that penetrate the Earth as body waves. Each wave has a characteristic speed and style of motion.

- **P-wave / Primary wave:** The primary wave the first seismic wave detected by seismographs due to high velocity in the rock environment between 5 and 8 km/s. The wave is usually relatively low-amplitude, does not carry much of the earthquake energy, and thus does not cause a significant damage.
- **S-wave / Secondary wave:** The secondary wave travels slower than the P-wave (3 to 5 km/s). However, it carries more energy and causes more damage than the P-wave.

EEW systems need to reliably detect the P-waves as this will allow for faster detections and therefore provide more time for the end users to take action. There is a number of traditional seismological algorithms that are usually based on rapid increase of [amplitude of ground motion](https://www.esgsolutions.com/technical-resources/microseismic-knowledgebase/event-detection-and-triggering). They are simple and effective, yet, they lack the ability to distinguish between signals created by earthquakes and other kind of disturbances (slamming doors, passing trucks..). In recent years, seismologists and data scientist have started to utilize neural-networks-based algorithms, hoping that those will be able to reduce the number of false positive detections.
