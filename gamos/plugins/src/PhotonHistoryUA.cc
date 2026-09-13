#include "PhotonHistoryUA.hh"

#include "G4Event.hh"
#include "G4Gamma.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

PhotonHistoryUA::PhotonHistoryUA(const G4String& outputFile) {
  fOut.open(outputFile);
  fOut << "event_id,energy_keV,x_mm,y_mm,n_compton_history,origin_volume,n_steps\n";
}

PhotonHistoryUA::~PhotonHistoryUA() {
  if (fOut.is_open()) fOut.close();
}

void PhotonHistoryUA::PreUserTrackingAction(const G4Track* track) {
  const G4int id = track->GetTrackID();
  const G4int parent = track->GetParentID();

  if (parent == 0) {  // primary decay photon
    fScatterOrder[id] = 0;
    fOriginVolume[id] =
        track->GetVolume() ? track->GetVolume()->GetName() : G4String("source");
    return;
  }

  // Inherit the parent's history. A photon created by a process other than
  // Compton scattering (fluorescence, bremsstrahlung) is not a primary: it
  // carries whatever history its parent had, so it can never be counted as an
  // unscattered photon.
  const auto it = fScatterOrder.find(parent);
  fScatterOrder[id] = (it != fScatterOrder.end()) ? it->second : 0;

  const auto origin = fOriginVolume.find(parent);
  fOriginVolume[id] =
      (origin != fOriginVolume.end()) ? origin->second : G4String("unknown");
}

void PhotonHistoryUA::BeginOfEventAction(const G4Event* event) {
  fEventId = event->GetEventID();
  fEnergy = fWeightedX = fWeightedY = 0.;
  fSteps = 0;
  fEnergyByScatterOrder.clear();
  fScatterOrder.clear();
  fOriginVolume.clear();
  fOrigin = "unknown";
}

void PhotonHistoryUA::UserSteppingAction(const G4Step* step) {
  const G4Track* track = step->GetTrack();
  const G4int id = track->GetTrackID();

  // Compton scattering keeps the photon's track ID, so the counter belongs to
  // this track and is inherited by every secondary it later produces.
  const G4VProcess* process = step->GetPostStepPoint()->GetProcessDefinedStep();
  if (process && process->GetProcessName() == "compt" &&
      track->GetDefinition() == G4Gamma::Definition()) {
    fScatterOrder[id] += 1;
  }

  const G4double edep = step->GetTotalEnergyDeposit();
  if (edep <= 0.) return;

  const auto* touchable = step->GetPreStepPoint()->GetTouchableHandle()();
  if (!touchable || !touchable->GetVolume()) return;
  if (touchable->GetVolume()->GetName() != "crystal") return;

  const G4ThreeVector position = step->GetPreStepPoint()->GetPosition();
  fEnergy += edep;
  fWeightedX += edep * position.x();
  fWeightedY += edep * position.y();
  fSteps += 1;
  fEnergyByScatterOrder[fScatterOrder[id]] += edep;
  if (fOriginVolume.count(id)) fOrigin = fOriginVolume[id];
}

void PhotonHistoryUA::EndOfEventAction(const G4Event*) {
  if (fEnergy <= 0.) return;  // no detection in this event

  // Classify the event by the photon history that deposited most of its
  // energy, rather than by whichever step happened to be last.
  G4int dominantOrder = 0;
  G4double best = -1.;
  for (const auto& entry : fEnergyByScatterOrder) {
    if (entry.second > best) {
      best = entry.second;
      dominantOrder = entry.first;
    }
  }

  fOut << fEventId << ',' << fEnergy / keV << ',' << (fWeightedX / fEnergy) / mm << ','
       << (fWeightedY / fEnergy) / mm << ',' << dominantOrder << ',' << fOrigin << ','
       << fSteps << '\n';
}
