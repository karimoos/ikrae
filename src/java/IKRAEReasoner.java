// src/java/IKRAEReasoner.java

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.HermiT.ReasonerFactory;

import java.util.*;

public class IKRAEReasoner {

    private OWLOntology ontology;
    private OWLReasoner reasoner;
    private OWLOntologyManager manager;
    private OWLDataFactory factory;

    // Ontology IRIs
    private static final String BASE = "http://ikrae.org/ontology#";
    private static final IRI LO_CLASS = IRI.create(BASE + "LearningObject");
    private static final IRI INF_CLASS = IRI.create(BASE + "Infeasible");

    public boolean loadOntology(String filePath) {
        try {
            manager = OWLManager.createOWLOntologyManager();
            ontology = manager.loadOntologyFromOntologyDocument(new java.io.File(filePath));
            factory = manager.getOWLDataFactory();

            // Create the HermiT reasoner
            reasoner = new ReasonerFactory().createReasoner(ontology);

            // Precompute inferences
            reasoner.precomputeInferences();

            System.out.println("[IKRAEReasoner] Ontology loaded and reasoner initialized.");
            return true;

        } catch (Exception e) {
            System.err.println("[IKRAEReasoner] ERROR loading ontology:");
            e.printStackTrace();
            return false;
        }
    }

    public Map<String, Object> runReasoningAndFilter() {

        Map<String, Object> result = new HashMap<>();
        List<String> feasible = new ArrayList<>();
        List<Map<String, Object>> explanations = new ArrayList<>();

        // LearningObject class
        OWLClass loClass = factory.getOWLClass(LO_CLASS);
        OWLClass infeasibleClass = factory.getOWLClass(INF_CLASS);

        // Get all individuals of LearningObject
        Set<OWLNamedIndividual> los =
                reasoner.getInstances(loClass, false).getFlattened();

        for (OWLNamedIndividual lo : los) {
            String id = lo.getIRI().getShortForm();

            boolean isInfeasible = reasoner.isEntailed(
                    factory.getOWLClassAssertionAxiom(infeasibleClass, lo)
            );

            if (!isInfeasible) {
                feasible.add(id);
            } else {
                explanations.add(
                    Map.of(
                        "lo_id", id,
                        "reason", "SWRL constraint marked object as Infeasible"
                    )
                );
            }
        }

        result.put("feasible", feasible);
        result.put("explanations", explanations);
        return result;
    }
}
