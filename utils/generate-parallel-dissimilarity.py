if __name__ == "__main__":
    animals = []
    with open("animal_list.txt") as handle:
        for line in handle:
            animals.append(line.strip())

    with open("parallel_dissimilarity.txt", "w") as handle:
        for (i, animal1) in enumerate(animals):
            for (j, animal2) in enumerate(animals):
                if j < i:
                    continue
                handle.write(f"python3 brain-alignment.py -a1 {animal1} -a2 {animal2} -s\n")
