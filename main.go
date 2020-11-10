package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	tele "gopkg.in/tucnak/telebot.v3"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

const (
	graphFile  = "./model/tensorflow_inception_graph.pb"
	labelsFile = "./model/imagenet_comp_graph_label_strings.txt"
)

// Label type
type Label struct {
	Label       string  `json:"label"`
	Probability float32 `json:"probability"`
}
type FileApi struct {
	Ok     bool `json:"ok"`
	Result struct {
		FileID       string `json:"file_id"`
		FileUniqueID string `json:"file_unique_id"`
		FileSize     int    `json:"file_size"`
		FilePath     string `json:"file_path"`
	} `json:"result"`
}

// Labels type
type Labels []Label

func (a Labels) Len() int           { return len(a) }
func (a Labels) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a Labels) Less(i, j int) bool { return a[i].Probability > a[j].Probability }

var (
	modelGraph *tensorflow.Graph
	labels     []string
	apiKey     string
)

func init() {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")
	var err error
	modelGraph, labels, err = loadModel()
	if err != nil {
		log.Fatalf("unable to load model: %v", err)
	}
	apiKey = "724608608:AAGs-qqTYMLUdTRq_GdJHNPh-zoZ2A8kU9U"
	// HandlePicture(nil, modelGraph, labels)
}

func main() {
	// file, err := os.Open("AQADMTxHmC4AA0jLAgAB.jpg")
	// file, err := download("photos/file_9.jpg", "AQAD02BQmC4AA93BAgAB")
	// nfile, err := os.Open(file.Name())
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// defer file.Close()
	// err = HandlePicture(nfile)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	botInit()
}

//HandlePicture handle picture and normilize it
func HandlePicture(file *os.File) (Labels, error) {
	// Get normalized tensor
	tensor, err := normalizeImage(file)
	if err != nil {
		// log.Fatalf("unable to make a tensor from image: %v", err)
		return nil, err
	}

	// Create a session for inference over modelGraph
	session, err := tensorflow.NewSession(modelGraph, nil)
	if err != nil {
		// log.Fatalf("unable to make a tensor from image: %v", err)
		return nil, err
	}

	output, err := session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			modelGraph.Operation("input").Output(0): tensor,
		},
		[]tensorflow.Output{
			modelGraph.Operation("output").Output(0),
		},
		nil)
	if err != nil {
		// log.Fatalf("unable to make a tensor from image: %v", err)
		return nil, err
	}

	res := getTopFiveLabels(labels, output[0].Value().([][]float32)[0])
	for _, l := range res {
		fmt.Printf("label: %s, probability: %.2f%%\n", l.Label, l.Probability*100)
	}
	return res, nil
}

func botInit() {
	b, err := tele.NewBot(tele.Settings{
		Token:  apiKey,
		Poller: &tele.LongPoller{Timeout: 10 * time.Second},
	})

	if err != nil {
		return
	}
	b.Handle(tele.OnPhoto, func(c tele.Context) error {
		fmt.Println(c.Chat().Username)
		fmt.Println("Photo")
		m := c.Message()
		fileID := m.Photo.FileID
		fmt.Println("File Id: %v", fileID)
		fileURL := fmt.Sprintf("https://api.telegram.org/bot724608608:AAGs-qqTYMLUdTRq_GdJHNPh-zoZ2A8kU9U/getFile?file_id=%v", fileID)
		fmt.Println("File url: %v", fileURL)
		resp, err := http.Get(fileURL)
		if err != nil {
			log.Fatal(err)
		}
		defer resp.Body.Close()
		buf := new(bytes.Buffer)
		buf.ReadFrom(resp.Body)
		bufString := buf.String()
		fmt.Println("Buf string: %v", bufString)
		fileApi := FileApi{}
		json.Unmarshal(buf.Bytes(), &fileApi)
		fmt.Println(fileApi)
		file, err := download(fileApi.Result.FilePath, fileApi.Result.FileUniqueID)
		if err != nil {
			return err
		}
		defer file.Close()
		nfile, err := os.Open(file.Name())
		if err != nil {
			return err
		}
		var finalStrs []string
		res, err := HandlePicture(nfile)
		os.Remove(file.Name())
		for _, l := range res {
			str := fmt.Sprintf("label: %s, probability: %.2f%%\n", l.Label, l.Probability*100)
			finalStrs = append(finalStrs, str)

		}
		c.Reply(strings.Join(finalStrs, ""))
		if err != nil {
			return err
		}

		return nil
	})
	b.Start()
}

func download(path string, id string) (*os.File, error) {
	fileName := fmt.Sprintf("%v.jpg", id)
	file, err := os.Create(fileName)
	if err != nil {
		return nil, err
	}
	newPath := fmt.Sprintf("https://api.telegram.org/file/bot724608608:AAGs-qqTYMLUdTRq_GdJHNPh-zoZ2A8kU9U/%v", path)
	resp, err := http.Get(newPath)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	buf.WriteTo(file)
	return file, nil
}

func getImage(filePath string, id string) (io.ReadCloser, error) {
	file, err := os.Create("file.jpg")
	if err != nil {
		return nil, err
	}

	resp, err := http.Get(fmt.Sprintf("https://api.telegram.org/file/bot724608608:AAGs-qqTYMLUdTRq_GdJHNPh-zoZ2A8kU9U/%v", filePath))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	buf.WriteTo(file)
	return file, nil
}

func loadModel() (*tensorflow.Graph, []string, error) {
	// Load inception model
	model, err := ioutil.ReadFile(graphFile)
	if err != nil {
		return nil, nil, err
	}
	graph := tensorflow.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return nil, nil, err
	}

	// Load labels
	labelsFile, err := os.Open(labelsFile)
	if err != nil {
		return nil, nil, err
	}
	defer labelsFile.Close()
	scanner := bufio.NewScanner(labelsFile)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}

	return graph, labels, scanner.Err()
}

func getTopFiveLabels(labels []string, probabilities []float32) []Label {
	var resultLabels []Label
	for i, p := range probabilities {
		if i >= len(labels) {
			break
		}
		resultLabels = append(resultLabels, Label{Label: labels[i], Probability: p})
	}

	sort.Sort(Labels(resultLabels))
	return resultLabels[:5]
}

func normalizeImage(body io.ReadCloser) (*tensorflow.Tensor, error) {
	var buf bytes.Buffer
	io.Copy(&buf, body)

	tensor, err := tensorflow.NewTensor(buf.String())
	if err != nil {
		return nil, err
	}

	graph, input, output, err := getNormalizedGraph()
	if err != nil {
		return nil, err
	}

	session, err := tensorflow.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}

	normalized, err := session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			input: tensor,
		},
		[]tensorflow.Output{
			output,
		},
		nil)
	if err != nil {
		return nil, err
	}

	return normalized[0], nil
}

// Creates a graph to decode, rezise and normalize an image
func getNormalizedGraph() (graph *tensorflow.Graph, input, output tensorflow.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tensorflow.String)
	// 3 return RGB image
	decode := op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))

	// Sub: returns x - y element-wise
	output = op.Sub(s,
		// make it 224x224: inception specific
		op.ResizeBilinear(s,
			// inserts a dimension of 1 into a tensor's shape.
			op.ExpandDims(s,
				// cast image to float type
				op.Cast(s, decode, tensorflow.Float),
				op.Const(s.SubScope("make_batch"), int32(0))),
			op.Const(s.SubScope("size"), []int32{224, 224})),
		// mean = 117: inception specific
		op.Const(s.SubScope("mean"), float32(117)))
	graph, err = s.Finalize()

	return graph, input, output, err
}
