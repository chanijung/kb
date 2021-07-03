
import torch
import numpy as np
import h5py

from allennlp.models.archival import load_archive
from kb.common import JsonFile
import simplejson as json
import kb.kg_embedding
from nltk.corpus import wordnet as wn



# includes @@PADDING@@, @@UNKNOWN@@, @@MASK@@, @@NULL@@
NUM_EMBEDDINGS = 117663

def generate_wordnet_synset_vocab(entity_file, vocab_file):
    vocab = ['@@UNKNOWN@@']
    
    with JsonFile(entity_file, 'r') as fin:
        for node in fin:
            if node['type'] == 'synset':
                vocab.append(node['id'])

    vocab.append('@@MASK@@')
    vocab.append('@@NULL@@')

    with open(vocab_file, 'w') as fout:
        fout.write('\n'.join(vocab))


def extract_tucker_embeddings(tucker_archive, vocab_file, tucker_hdf5):
    archive = load_archive(tucker_archive)

    with open(vocab_file, 'r') as fin:
        vocab_list = fin.read().strip().split('\n')

    # get embeddings
    embed = archive.model.kg_tuple_predictor.entities.weight.detach().numpy()
    print(f'embed.shape {embed.shape}')
    # print(embed[0])
    out_embeddings = np.zeros((NUM_EMBEDDINGS, embed.shape[1]))
    
    print(f'out_embeddings.shape {out_embeddings.shape}')

    vocab = archive.model.vocab


    for k, entity in enumerate(vocab_list):
        embed_id = vocab.get_token_index(entity, 'entity')
        if entity in ('@@MASK@@', '@@NULL@@'):
            # these aren't in the tucker vocab -> random init
            out_embeddings[k + 1, :] = np.random.randn(1, embed.shape[1]) * 0.004
        elif entity != '@@UNKNOWN@@':
            assert embed_id != 1
            # k = 0 is @@UNKNOWN@@, and want it at index 1 in output
            out_embeddings[k + 1, :] = embed[embed_id, :]

    print(out_embeddings[0])
    # write out to file
    with h5py.File(tucker_hdf5, 'w') as fout:
        ds = fout.create_dataset('tucker', data=out_embeddings)


def debias_tucker_embeddings(tucker_archive, tucker_hdf5, vocab_file):
    #Get tucker embeddings numpy array
    with h5py.File(tucker_hdf5, 'r') as fin:
        tucker = fin['tucker'][...]

    #Get list of words in vocabulary
    with open(vocab_file, 'r') as fin:
        vocab_list = fin.read().strip().split('\n')

    #Extract gender directional vectors
    archive = load_archive(tucker_archive)
    vocab = archive.model.vocab
    gender_dir_vecs = []
    for info in ["n.01", "n.02", "a.01", "s.02", "s.03"]:
        id1 = vocab.get_token_index('female.'+info, 'entity')
        id2 = vocab.get_token_index('male.'+info, 'entity')
        gender_dir_vecs.append(tucker[id1+1, :]-tucker[id2+1,:])
    lambdas = [0.5]*5   #Parameters which decide the amount of debiasing

    #Debias tucker embeddings of occupation words
    occ = open("bin/professions.txt", "r")
    num_debiased_word = 0
    num_debiased_embeddings = 0
    for line in occ.readlines():
        entities = [w for w in vocab_list if w.startswith(line.strip()+".n.")]
        if len(entities)>0:
            num_debiased_word += 1
        for entity in entities: #For each entity corresponding to each occupation
            id = vocab.get_token_index(entity, 'entity')
            # print(f'entity {entity}, id {id}')
            if id<0 or id>=NUM_EMBEDDINGS:
                continue
            num_debiased_embeddings += 1
            for i in range(len(gender_dir_vecs)):  #Debias the embedding
                gdv = gender_dir_vecs[i]
                lam = lambdas[i]
                tucker[id+1, :] = tucker[id+1, :] - lam * np.dot(tucker[id+1, :], gdv) / np.linalg.norm(gdv)
    print(f'debiased {num_debiased_word} words, {num_debiased_embeddings} embeddings')

    #Write to the new embeddings file.
    with h5py.File('bin/debiased_tucker_embeddings.hdf5', 'w') as fout:
        ds = fout.create_dataset('tucker', data=tucker)
            


def get_gensen_synset_definitions(entity_file, vocab_file, gensen_file):
    from gensen import GenSen, GenSenSingle

    gensen_1 = GenSenSingle(
        model_folder='./data/models',
        filename_prefix='nli_large_bothskip',
        pretrained_emb='./data/embedding/glove.840B.300d.h5'
    )
    gensen_1.eval()

    definitions = {}
    with open(entity_file, 'r') as fin:
        for line in fin:
            node = json.loads(line)
            if node['type'] == 'synset':
                definitions[node['id']] = node['definition']

    with open(vocab_file, 'r') as fin:
        vocab_list = fin.read().strip().split('\n')

    # get the descriptions
    sentences = [''] * NUM_EMBEDDINGS
    for k, entity in enumerate(vocab_list):
        definition = definitions.get(entity)
        if definition is None:
            assert entity in ('@@UNKNOWN@@', '@@MASK@@', '@@NULL@@')
        else:
            sentences[k + 1] = definition

    embeddings = np.zeros((NUM_EMBEDDINGS, 2048), dtype=np.float32)
    for k in range(0, NUM_EMBEDDINGS, 32):
        sents = sentences[k:(k+32)]
        reps_h, reps_h_t = gensen_1.get_representation(
            sents, pool='last', return_numpy=True, tokenize=True
        )
        embeddings[k:(k+32), :] = reps_h_t
        print(k)

    with h5py.File(gensen_file, 'w') as fout:
        ds = fout.create_dataset('gensen', data=embeddings)


def combine_tucker_gensen(tucker_hdf5, gensen_hdf5, all_file):
    with h5py.File(tucker_hdf5, 'r') as fin:
        tucker = fin['tucker'][...]

    with h5py.File(gensen_hdf5, 'r') as fin:
        gensen = fin['gensen'][...]

    all_embeds = np.concatenate([tucker, gensen], axis=1)
    all_e = all_embeds.astype(np.float32)

    with h5py.File(all_file, 'w') as fout:
        ds = fout.create_dataset('tucker_gensen', data=all_e)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_wordnet_synset_vocab', default=False, action="store_true")
    parser.add_argument('--entity_file', type=str)
    parser.add_argument('--vocab_file', type=str)

    parser.add_argument('--generate_gensen_embeddings', default=False, action="store_true")
    parser.add_argument('--gensen_file', type=str)

    parser.add_argument('--extract_tucker', default=False, action="store_true")
    parser.add_argument('--tucker_archive_file', type=str)
    parser.add_argument('--tucker_hdf5_file', type=str)

    parser.add_argument('--debias_tucker', default=False, action="store_true")

    parser.add_argument('--combine_tucker_gensen', default=False, action="store_true")
    parser.add_argument('--all_embeddings_file', type=str)

    args = parser.parse_args()


    if args.generate_wordnet_synset_vocab:
        generate_wordnet_synset_vocab(args.entity_file, args.vocab_file)
    elif args.generate_gensen_embeddings:
        get_gensen_synset_definitions(args.entity_file, args.vocab_file, args.gensen_file)
    elif args.extract_tucker:
        extract_tucker_embeddings(args.tucker_archive_file, args.vocab_file, args.tucker_hdf5_file)
    elif args.combine_tucker_gensen:
        combine_tucker_gensen(args.tucker_hdf5_file, args.gensen_file, args.all_embeddings_file)
    elif args.debias_tucker:
        debias_tucker_embeddings(args.tucker_archive_file, args.tucker_hdf5_file, args.vocab_file)
    else:
        raise ValueError

