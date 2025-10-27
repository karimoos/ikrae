// src/java/IKRAEReasoner.java
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.HermiT.Reasoner;
import java.util.*;

public class IKRAEReasoner {
    private OWLOntology ontology;
    private OWLReasoner reasoner;
    private OWLOntologyManager manager;
    private OWLDataFactory factory;

    public boolean loadOntology(String filePath) {
        try {
            manager = OWLManager.createOWLOntologyManager();
            ontology = manager.loadOntologyFromOntologyDocument(new java.io.File(filePath));
            reasoner = new Reasoner(ontology);
            factory = manager.getOWLDataFactory();
            return true;
        } catch (Exception e) { e.printStackTrace(); return false; }
    }

    public ArrayList runReasoningAndFilter() {
        ArrayList result = new ArrayList();
        ArrayList<String> feasible = new ArrayList<>();
        ArrayList<Map<String, Object>> explanations = new ArrayList<>();

        Set<OWLNamedIndividual> los = reasoner.getInstances(
            factory.getOWLClass(IRI.create("http://ikrae.org/ontology#LearningObject")), false
        ).getFlattened();

        for (OWLNamedIndividual lo : los) {
            String id = lo.getIRI().getShortForm();
            boolean infeasible = reasoner.isEntailed(factory.getOWLClassAssertionAxiom(
                factory.getOWLClass(IRI.create("http://ikrae.org/ontology#Infeasible")), lo
            ));
            if (!infeasible) feasible.add(id);
            else explanations.add(Map.of("lo_id", id, "reason", "SWRL violation"));
        }
        result.add(feasible);
        result.add(explanations);
        return result;
    }
}
