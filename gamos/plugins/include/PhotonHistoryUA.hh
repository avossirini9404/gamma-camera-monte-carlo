// Photon lineage tracking and event-level detection scoring.
//
// Two things this class exists to get right, both of which the previous
// step-level implementation got wrong.
//
// 1. Lineage. Compton scattering in Geant4 does *not* create a new photon
//    track: the incident gamma keeps its track ID with reduced energy, so
//    counting "compt" on the photon track is correct for that process alone.
//    It is not correct for the photons that other processes create — lead
//    K X-rays following photoelectric absorption in the septa, bremsstrahlung,
//    fluorescence — which start as fresh tracks and would otherwise be scored
//    as primaries. Nor is it correct for the secondary electrons that actually
//    deposit the energy in the crystal, whose own track ID carries no scatter
//    history at all. PreUserTrackingAction therefore propagates the scatter
//    order from parent to child, so every depositing track knows the history
//    of the photon it came from.
//
// 2. Event-level scoring. A camera integrates the scintillation light of a
//    whole interaction and reports one event, with one energy and one
//    position. Geant4 deposits that energy over several steps. Energy is
//    accumulated per event and written once, with the energy-weighted
//    centroid as the position, which is what an Anger position circuit
//    estimates.
#ifndef PhotonHistoryUA_hh
#define PhotonHistoryUA_hh

#include <fstream>
#include <map>
#include <string>

#include "G4UserEventAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserTrackingAction.hh"

class PhotonHistoryUA : public G4UserSteppingAction,
                        public G4UserEventAction,
                        public G4UserTrackingAction {
 public:
  explicit PhotonHistoryUA(const G4String& outputFile);
  ~PhotonHistoryUA() override;

  void PreUserTrackingAction(const G4Track* track) override;
  void UserSteppingAction(const G4Step* step) override;
  void BeginOfEventAction(const G4Event* event) override;
  void EndOfEventAction(const G4Event* event) override;

 private:
  // Per-track scatter order, inherited from the parent track at creation.
  std::map<G4int, G4int> fScatterOrder;
  std::map<G4int, G4String> fOriginVolume;

  // Per-event accumulators, reset at BeginOfEventAction.
  G4double fEnergy{0.};
  G4double fWeightedX{0.};
  G4double fWeightedY{0.};
  G4int fSteps{0};
  std::map<G4int, G4double> fEnergyByScatterOrder;
  G4String fOrigin{"unknown"};

  std::ofstream fOut;
  G4int fEventId{0};
};

#endif
