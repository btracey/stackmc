// package fold implements types for generating Folds for running StackMC analysis.
package fold

import (
	"math/rand"

	"github.com/btracey/stackmc"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

func addSamples(samps mat64.Dense, nNew, dim int, dist stackmc.Distribution) (updatedSamps mat64.Dense) {
	//rand.Seed(0)
	//fmt.Println("fixed seed")
	newSamps := mat64.NewDense(nNew, dim, nil)
	dist.Sample(newSamps)
	r, _ := samps.Dims()
	if r == 0 {
		return *newSamps
	}
	oldsamps := samps
	samps = mat64.Dense{}
	samps.Stack(&oldsamps, newSamps)
	return samps
}

// FoldAdver is an advanced folder for research purposes. NSamples is the "real"
// samples from the function, and are indices 0 to nSamples - 1.
type AdvFolder interface {
	AdvFolds(nSamples, dim int, extraSampler stackmc.Distribution) (newSamples *mat64.Dense, folds []stackmc.Fold)
}

// Folder generates folds for the given number of samples.
type Folder interface {
	Folds(nSamples int)
}

func Partition(nData int, nFolds int) (training [][]int, testing [][]int) {
	if nFolds < 0 {
		panic("negative number of folds")
	}
	if nData < 0 {
		panic("negative amount of data")
	}
	if nFolds > nData {
		nFolds = nData
	}

	// Get a random permutation of the data samples
	perm := rand.Perm(nData)

	training = make([][]int, nFolds)
	testing = make([][]int, nFolds)

	nSampPerFold := nData / nFolds
	remainder := nData % nFolds

	idx := 0
	for i := 0; i < nFolds; i++ {
		nTestElems := nSampPerFold
		if i < remainder {
			nTestElems += 1
		}
		testing[i] = make([]int, nTestElems)
		copy(testing[i], perm[idx:idx+nTestElems])

		training[i] = make([]int, nData-nTestElems)
		copy(training[i], perm[:idx])
		copy(training[i][idx:], perm[idx+nTestElems:])

		idx += nTestElems
	}
	if idx != nData {
		panic("bad logic")
	}
	return
}

// All uses all of the samples for everything -- one fold.
type All struct{}

func (a All) Folds(nSamples int) []stackmc.Fold {
	folds := make([]stackmc.Fold, 1)
	folds[0].Train = make([]int, nSamples)
	folds[0].Update = make([]int, nSamples)
	folds[0].Assess = make([]int, nSamples)
	for i := range folds[0].Train {
		folds[0].Train[i] = i
		folds[0].Update[i] = i
		folds[0].Assess[i] = i
	}
	return folds
}

func (a All) AdvFolds(nSamples, dim int, es stackmc.Distribution) (*mat64.Dense, []stackmc.Fold) {
	folds := a.Folds(nSamples)
	return nil, folds
}

// KFold generates a set of StackMC folds generated from K-fold validation. The
// training data are the held-in data, while the Asses and Update are the same
// held-out data sets.
type KFold struct {
	K int
}

func (k KFold) Folds(nSamples int) []stackmc.Fold {
	training, testing := Partition(nSamples, k.K)
	folds := make([]stackmc.Fold, len(training))
	for i := range folds {
		folds[i].Train = training[i]
		folds[i].Update = testing[i]
		folds[i].Assess = testing[i]
	}
	return folds
}

func (k KFold) AdvFolds(nSamples, dim int, es stackmc.Distribution) (*mat64.Dense, []stackmc.Fold) {
	folds := k.Folds(nSamples)
	return nil, folds
}

type MultiKFold struct {
	K     int
	Multi int
}

func (m MultiKFold) Folds(nSamples int) []stackmc.Fold {
	var folds []stackmc.Fold
	for i := 0; i < m.Multi; i++ {
		folds = append(folds, KFold{m.K}.Folds(nSamples)...)
	}
	return folds
}

func (m MultiKFold) AdvFolds(nSamples, dim int, es stackmc.Distribution) (*mat64.Dense, []stackmc.Fold) {
	folds := m.Folds(nSamples)
	return nil, folds
}

type KFoldUpdateAll struct {
	K      int
	Update bool
	Assess bool
}

func (k KFoldUpdateAll) AdvFolds(nSamples, dim int, es stackmc.Distribution) (*mat64.Dense, []stackmc.Fold) {
	folds := KFold{k.K}.Folds(nSamples)
	for i := range folds {
		if k.Update {
			folds[i].Update = make([]int, nSamples)
			for j := range folds[i].Update {
				folds[i].Update[j] = j
			}
		}
		if k.Assess {
			folds[i].Assess = make([]int, nSamples)
			for j := range folds[i].Assess {
				folds[i].Assess[j] = j
			}
		}
	}
	return nil, folds
}

// KFoldScramble is like KFold except the training indices are scrambeled and
// such.
type KFoldScramble struct {
	K int
}

func (k KFoldScramble) Folds(nSamples int) []stackmc.Fold {
	folds := KFold{k.K}.Folds(nSamples)
	// Lazy way, just have K-1 copies of every index, and put them randomly.
	var inds []int
	for i := 0; i < k.K-1; i++ {
		for j := 0; j < nSamples; j++ {
			inds = append(inds, j)
		}
	}
	perm := rand.Perm(len(inds))
	var count int
	for i := range folds {
		for j := range folds[i].Train {
			folds[i].Train[j] = inds[perm[count]]
			count++
		}
	}
	if count != len(perm) {
		panic("bad count")
	}
	return folds
}

func (k KFoldScramble) AdvFolds(nSamples, dim int, es stackmc.Distribution) (*mat64.Dense, []stackmc.Fold) {
	folds := k.Folds(nSamples)
	return nil, folds
}

type KFoldAlphaCorrect struct {
	K         int
	SampMul   float64
	IndUpdate bool
}

func (k KFoldAlphaCorrect) AdvFolds(nSamples, dim int, es stackmc.Distribution) (*mat64.Dense, []stackmc.Fold) {
	folds := KFold{k.K}.Folds(nSamples)

	idx := nSamples
	var samps mat64.Dense
	for _, fold := range folds {
		nNew := int(k.SampMul * float64(len(fold.Assess)))
		newsamp := mat64.NewDense(nNew, dim, nil)
		es.Sample(newsamp)
		r, _ := samps.Dims()
		if r == 0 {
			samps = *newsamp
		} else {
			oldsamps := samps
			samps = mat64.Dense{}
			samps.Stack(&oldsamps, newsamp)
		}
		fold.Assess = make([]int, nNew)
		for j := range fold.Assess {
			fold.Assess[j] = idx
			idx++
		}

		if k.IndUpdate {
			nNew = len(fold.Update)
			newsamp = mat64.NewDense(nNew, dim, nil)
			es.Sample(newsamp)
			oldsamps := samps
			samps = mat64.Dense{}
			samps.Stack(&oldsamps, newsamp)
			for j := range fold.Update {
				fold.Update[j] = idx
				idx++
			}
		}
	}
	return &samps, folds
}

/*
// k fold, except sample without replacement on the in samples
type KFoldBootstrap struct {
	K int
}

func (k KFoldBootstrap) Folds(nSamples int) []stackmc.Fold {
	folds := KFold{k.K}.Folds(nSamples)
	for i := range folds {
		perm := rand.Perm(nSamples)
		copy(folds[i].Train, perm)
	}
	return folds
}

func (k KFoldBootstrap) AdvFolds(nSamples, dim int, es stackmc.Distribution) (*mat64.Dense, []stackmc.Fold) {
	folds := k.Folds(nSamples)
	return nil, folds
}
*/

// KFoldBoot is like the Bootstrap, except the statistics of the k folds are
// all represented (each training point is in n-1 of the sets)
type KFoldBoot struct {
	K        int
	Multi    int
	IndAlpha bool
}

func (k KFoldBoot) AdvFolds(nSamples, dim int, es stackmc.Distribution) (*mat64.Dense, []stackmc.Fold) {
	folds := k.Folds(nSamples)
	return nil, folds
}

func (k KFoldBoot) Folds(nSamples int) []stackmc.Fold {
	var folds []stackmc.Fold
	for i := 0; i < k.Multi; i++ {
		// Generate two independent k-folds, then push the testing and training
		// together.
		tr1, _ := Partition(nSamples, k.K)
		_, te2 := Partition(nSamples, k.K)
		_, te3 := Partition(nSamples, k.K)
		var fold stackmc.Fold
		for j := 0; j < len(tr1); j++ {
			fold.Train = tr1[j]
			fold.Update = te2[j]
			if k.IndAlpha {
				fold.Assess = te3[j]
			} else {
				fold.Assess = te2[j]
			}
			folds = append(folds, fold)
		}
	}
	return folds
}

// KFoldInd is the same as KFold, except independent samples are generated for
// assess and/or train.
type KFoldInd struct {
	K            int
	Train        bool
	TrainMul     float64
	TrainSame    bool
	Assess       bool
	AssessSame   bool
	AssessMul    float64
	AssessUseAll bool
	Update       bool
	UpdateUseAll bool

	AssessUpdateSame bool
}

func (k KFoldInd) AdvFolds(nSamples, dim int, es stackmc.Distribution) (*mat64.Dense, []stackmc.Fold) {
	folds := KFold{k.K}.Folds(nSamples)
	idx := nSamples
	var samps mat64.Dense

	if k.TrainSame {
		if !k.Train {
			panic("bad case")
		}
		nNew := int(float64(nSamples) * k.TrainMul)
		samps = addSamples(samps, nNew, dim, es)
		for i := range folds {
			folds[i].Train = make([]int, nNew)
		}
		for j := 0; j < nNew; j++ {
			for i := range folds {
				folds[i].Train[j] = idx
			}
			idx++
		}
		//fmt.Println("samps = ", samps)
	}

	if k.AssessSame {
		if !k.Assess {
			panic("bad case")
		}
		nNew := int(float64(nSamples) * k.AssessMul)
		samps = addSamples(samps, nNew, dim, es)
		for i := range folds {
			folds[i].Assess = make([]int, nNew)
		}
		for j := 0; j < nNew; j++ {
			for i := range folds {
				folds[i].Assess[j] = idx
			}
			idx++
		}
	}

	for i := range folds {
		if k.Train && !k.TrainSame {
			nNew := int(float64(len(folds[i].Train)) * k.TrainMul)
			samps = addSamples(samps, nNew, dim, es)
			for j := range folds[i].Train {
				folds[i].Train[j] = idx
				idx++
			}
		}
		if k.AssessUpdateSame {
			if k.Update == false || k.Assess == false || k.AssessMul != 0 || k.UpdateUseAll == true || k.AssessSame || k.AssessUseAll {
				panic("bad case")
			}
			nNew := len(folds[i].Assess)
			samps = addSamples(samps, nNew, dim, es)
			for j := range folds[i].Assess {
				folds[i].Assess[j] = idx
				folds[i].Update[j] = idx
				idx++
			}
			continue
		}
		if k.AssessMul != 0 && !k.Assess {
			panic("bad case")
		}
		if k.AssessUseAll && !k.Assess {
			folds[i].Assess = make([]int, nSamples)
			for j := range folds[i].Assess {
				folds[i].Assess[j] = j
			}
		}
		if k.Assess && !k.AssessSame {
			if k.AssessUseAll {
				panic("bad case")
			}
			if k.AssessMul == 0 {
				panic("bad case")
			}
			nNew := len(folds[i].Assess)
			if k.AssessMul != 0 {
				nNew = int(float64(nNew) * k.AssessMul)
			}
			samps = addSamples(samps, nNew, dim, es)
			folds[i].Assess = make([]int, nNew)
			for j := range folds[i].Assess {
				folds[i].Assess[j] = idx
				idx++
			}
		}
		if k.UpdateUseAll && !k.Update {
			folds[i].Update = make([]int, nSamples)
			for j := range folds[i].Update {
				folds[i].Update[j] = j
			}
		}
		if k.Update {
			if k.UpdateUseAll {
				panic("not coded")
			}
			nNew := len(folds[i].Update)
			samps = addSamples(samps, nNew, dim, es)
			for j := range folds[i].Update {
				folds[i].Update[j] = idx
				idx++
			}
		}
	}
	return &samps, folds
}

// OneFold only does one computation. Always independent training samples. Possibly
// independent assessment samples. Update samples always the original samples.
type OneFold struct {
	TrainMul  float64
	Alpha     bool
	AlphaMul  float64
	AlphaSame bool
}

func (o OneFold) AdvFolds(nSamples, dim int, es stackmc.Distribution) (*mat64.Dense, []stackmc.Fold) {
	folds := make([]stackmc.Fold, 1)
	folds[0].Update = make([]int, nSamples)
	for i := range folds[0].Update {
		folds[0].Update[i] = i
	}
	idx := nSamples
	var samps mat64.Dense

	// Add independent training samples
	nNew := int(float64(nSamples) * o.TrainMul)
	samps = addSamples(samps, nNew, dim, es)
	folds[0].Train = make([]int, nNew)
	for i := range folds[0].Train {
		folds[0].Train[i] = idx
		idx++
	}

	if !o.Alpha {
		folds[0].Assess = make([]int, nSamples)
		for j := range folds[0].Assess {
			folds[0].Assess[j] = j
		}
		return &samps, folds
	}

	if o.AlphaSame {
		if o.AlphaMul != 1 {
			panic("bad setup")
		}
		folds[0].Assess = make([]int, nSamples)
		for i := range folds[0].Assess {
			folds[0].Assess[i] = i
		}
	}

	// Add independent alpha samples
	nNew = int(float64(nSamples) * o.AlphaMul)
	samps = addSamples(samps, nNew, dim, es)
	folds[0].Assess = make([]int, nNew)
	for i := range folds[0].Assess {
		folds[0].Assess[i] = idx
		idx++
	}
	return &samps, folds
}

type MultiBootstrap struct {
	K           int
	Times       int
	Replacement bool
}

func (m MultiBootstrap) Folds(nSamples int) []stackmc.Fold {
	var folds []stackmc.Fold
	for i := 0; i < m.Times; i++ {
		f := KFold{m.K}.Folds(nSamples)
		folds = append(folds, f...)
	}
	for i := range folds {
		if m.Replacement {
			for j := range folds[i].Train {
				folds[i].Train[j] = rand.Intn(nSamples)
			}
		} else {
			perm := rand.Perm(nSamples)
			copy(folds[i].Train, perm)
		}
	}
	return folds
}

func (m MultiBootstrap) AdvFolds(nSamples, dim int, es stackmc.Distribution) (*mat64.Dense, []stackmc.Fold) {
	folds := m.Folds(nSamples)
	return nil, folds
}

// Bootstrap does bootstrapping with replacement. Round.
type Bootstrap struct {
	// Insample and outsample always independently sampled.
	Replacement      bool
	AssessUpdateSame bool
	NumFolds         int
	TrainFraction    float64
	AssessFraction   float64
	UpdateFraction   float64
}

func (b Bootstrap) Folds(nSamples int) []stackmc.Fold {
	nTrainSamples := int(floats.Round(b.TrainFraction*float64(nSamples), 0))
	nAssessSamples := int(floats.Round(b.AssessFraction*float64(nSamples), 0))
	updateSamples := int(floats.Round(b.UpdateFraction*float64(nSamples), 0))

	var folds []stackmc.Fold
	for fold := 0; fold < b.NumFolds; fold++ {
		trainSamples := make([]int, nTrainSamples)
		assessSamples := make([]int, nAssessSamples)
		updateSamples := make([]int, updateSamples)
		if b.Replacement {
			for i := range trainSamples {
				trainSamples[i] = rand.Intn(nSamples)
			}
			for i := range assessSamples {
				assessSamples[i] = rand.Intn(nSamples)
			}
			if b.AssessUpdateSame {
				copy(updateSamples, assessSamples)
			} else {
				for i := range updateSamples {
					updateSamples[i] = rand.Intn(nSamples)
				}
			}
		} else {
			perm := rand.Perm(nSamples)
			copy(trainSamples, perm)
			perm = rand.Perm(nSamples)
			copy(assessSamples, perm)
			if b.AssessUpdateSame {
				copy(updateSamples, assessSamples)
			} else {
				perm = rand.Perm(nSamples)
				copy(updateSamples, perm)
			}
		}

		folds = append(folds,
			stackmc.Fold{
				Train:  trainSamples,
				Assess: assessSamples,
				Update: updateSamples,
			})
	}
	return folds
}

func (b Bootstrap) AdvFolds(nSamples, dim int, es stackmc.Distribution) (*mat64.Dense, []stackmc.Fold) {
	folds := b.Folds(nSamples)
	return nil, folds
}

type BootstrapInAllOut struct {
	Multi int
}

func (b BootstrapInAllOut) Folds(nSamples int) []stackmc.Fold {
	var folds []stackmc.Fold
	for f := 0; f < b.Multi; f++ {
		var fold stackmc.Fold
		fold.Train = make([]int, nSamples)
		for i := range fold.Train {
			fold.Train[i] = rand.Intn(nSamples)
		}
		fold.Assess = make([]int, nSamples)
		fold.Update = make([]int, nSamples)
		for i := range fold.Assess {
			fold.Assess[i] = i
			fold.Update[i] = i
		}
		folds = append(folds, fold)
	}
	return folds
}

func (b BootstrapInAllOut) AdvFolds(nSamples, dim int, es stackmc.Distribution) (*mat64.Dense, []stackmc.Fold) {
	folds := b.Folds(nSamples)
	return nil, folds
}
