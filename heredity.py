import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def pass_one_gene(n):
    if n == 0:
        p = PROBS["mutation"]
    elif n == 1:
        p = 0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"]
    elif n == 2:
        p = 1 - PROBS["mutation"]
    return p


def pass_no_gene(n):
    if n == 0:
        p = 1 - PROBS["mutation"]
    elif n == 1:
        p = 0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"]
    elif n == 2:
        p = PROBS["mutation"]
    return p


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    p = 1.0  # Start with probability of 1

    for person in people:

        # mother
        m = people[person]["mother"]
        if m in one_gene:
            m_gene = 1
        elif m in two_genes:
            m_gene = 2
        else:
            m_gene = 0

        # father
        f = people[person]["father"]
        if f in one_gene:
            f_gene = 1
        elif f in two_genes:
            f_gene = 2
        else:
            f_gene = 0

        # one gene
        if person in one_gene:
            n_gene = 1
            if m == None and f == None:
                p *= PROBS["gene"][1]
            else:
                p *= pass_one_gene(m_gene) * pass_no_gene(f_gene) + \
                    pass_one_gene(f_gene) * pass_no_gene(m_gene)

        # two genes
        elif person in two_genes:
            n_gene = 2
            if m == None and f == None:
                p *= PROBS["gene"][2]
            else:
                p *= pass_one_gene(m_gene) * pass_one_gene(f_gene)

        # no gene
        else:
            n_gene = 0
            if m == None and f == None:
                p *= PROBS["gene"][0]
            else:
                p *= pass_no_gene(m_gene) * pass_no_gene(f_gene)

        # trait
        if person in have_trait:
            p *= PROBS["trait"][n_gene][True]
        else:
            p *= PROBS["trait"][n_gene][False]
    return p


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        # Update gene distribution
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        # Update trait distribution
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        total = sum(probabilities[person]["gene"].values())
        for value in probabilities[person]["gene"]:
            probabilities[person]["gene"][value] /= total

        total = sum(probabilities[person]["trait"].values())
        for value in probabilities[person]["trait"]:
            probabilities[person]["trait"][value] /= total


def get_parent_genes(people, child, parent):
    """
    Get the gene counts of the given parent for the specified child.

    Args:
        people: Dictionary of people information.
        child: Name of the child person.
        parent: Name of the parent person (mother or father).

    Returns:
        A dictionary with "0", "1", and "2" keys representing the possible gene counts
        of the parent for the child. The values represent the probabilities of having those counts.
    """
    parent_genes = {
        0: 0,
        1: 0,
        2: 0
    }

    # If parent information is not available, use unconditional probabilities.
    if people[child][parent] is None:
        parent_genes = PROBS["gene"]
        return parent_genes

    # Get parent's gene count.
    parent_gene_count = people[people[child][parent]]["gene"]

    # Calculate probability of receiving each gene count from the parent.
    # Account for mutation probability here.
    for child_gene_count in range(3):
        if child_gene_count == 0:
            # Probability of child having 0 genes and receiving 0 or 1 from parent.
            parent_genes[0] += (
                PROBS["gene"][parent_gene_count] * (1 - PROBS["mutation"]) +
                PROBS["gene"][parent_gene_count - 1] * PROBS["mutation"]
            )
        elif child_gene_count == 1:
            # Probability of child having 1 gene and receiving 0 or 1 from parent.
            parent_genes[1] += (
                PROBS["gene"][parent_gene_count] * PROBS["mutation"] +
                PROBS["gene"][parent_gene_count - 1] * (1 - PROBS["mutation"])
            )
        else:
            # Probability of child having 2 genes and receiving 1 from parent.
            parent_genes[2] += PROBS["gene"][parent_gene_count] * \
                PROBS["mutation"]

    return parent_genes


def probability_of_child_genes(mother_genes, father_genes, child_gene_count):
    """
    Calculate the probability of a child having a specific gene count given
    gene counts of both parents.

    Args:
        mother_genes: Dictionary of probabilities for mother's gene counts.
        father_genes: Dictionary of probabilities for father's gene counts.
        child_gene_count: Desired gene count for the child (0, 1, or 2).

    Returns:
        The probability of the child having the specified gene count based on
        both parents' gene counts.
    """
    probability = 0

    # Iterate through all possible combinations of parent gene counts
    # and calculate the probability of each combination resulting in the
    # desired child gene count.
    for mother_gene in mother_genes:
        for father_gene in father_genes:
            # For each parent's possible gene count, check if their offspring
            # could have the desired child gene count.
            if (mother_gene + father_gene) == child_gene_count:
                # Combine the individual parent probabilities for this outcome.
                probability += mother_genes[mother_gene] * \
                    father_genes[father_gene]

    return probability


if __name__ == "__main__":
    main()
