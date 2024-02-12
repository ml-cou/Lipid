import { CircularProgress } from "@mui/material";
import { InputNumber, Radio } from "antd";
import React, { useState } from "react";
import toast, { Toaster } from "react-hot-toast";
import BeadsBondsInput from "./BeadsBondsInput";
import CompositionInput from "./CompositionInput";

const { Group: RadioGroup, Button: RadioButton } = Radio;

/**
 * Prediction is the main component that manages the state and logic of the application.
 * It handles the rendering of composition inputs, file uploads, and other data fields.
 */

function Prediction() {
  const [type, setType] = useState("single");
  const [data, setData] = useState({
    "Number of Water Molecules": "",
    "Salt (moles per liter)": 0.15,
    Temperature: 310,
    Pressure: 1,
    "Number of Lipid Per Layer": 2915,
    "Membrane Thickness": "",
    "Kappa BW-DCF(Bandwidth Dependent Die-electric Constant Fluctuation)": "",
    "Kappa-RSF": "",
  });
  const [compositions, setCompositions] = useState(initialCompositionState());
  const [predictionValue, setPredictionValue] = useState("");
  const [loading, setLoading] = useState(false);

  const [inputs, setInputs] = useState([
    {
      inputType: "upload",
      file: null,
      text: "",
      label: "Beads-Bonds Structure (Adjacency Matrix) for Lipid-1",
    },
    {
      inputType: "upload",
      file: null,
      text: "",
      label: "Beads Properties Structure (Node Feature Matrix) for Lipid-1",
    },
    {
      inputType: "upload",
      file: null,
      text: "",
      label: "Beads-Bonds Structure (Adjacency Matrix) for Lipid-2",
    },
    {
      inputType: "upload",
      file: null,
      text: "",
      label: "Beads Properties Structure (Node Feature Matrix) for Lipid-2",
    },
  ]);

  const setInputsType = (index, newType) => {
    setInputs((prev) =>
      prev.map((item, i) =>
        i === index ? { ...item, inputType: newType } : item
      )
    );
  };

  const handleFileChange = (index, newFile) => {
    setInputs((prev) =>
      prev.map((item, i) => (i === index ? { ...item, file: newFile } : item))
    );
  };

  const handleTextChange = (index, newText) => {
    setInputs((prev) =>
      prev.map((item, i) => (i === index ? { ...item, text: newText } : item))
    );
  };

  const handleTypeChange = (newType) => {
    setType(newType);
    setCompositions(compositionStateOnTypeChange(newType));
    setInputs([
      {
        inputType: "upload",
        file: null,
        text: "",
        label: "Beads-Bonds Structure (Adjacency Matrix) for Lipid-1",
      },
      {
        inputType: "upload",
        file: null,
        text: "",
        label: "Beads Properties Structure (Node Feature Matrix) for Lipid-1",
      },
      {
        inputType: "upload",
        file: null,
        text: "",
        label: "Beads-Bonds Structure (Adjacency Matrix) for Lipid-2",
      },
      {
        inputType: "upload",
        file: null,
        text: "",
        label: "Beads Properties Structure (Node Feature Matrix) for Lipid-2",
      },
    ]);
  };

  const handleCompositionChange = (id, field, value) => {
    console.log(id, field, value);
    setCompositions((prevCompositions) => {
      // Create a copy of the previous state
      const newCompositions = { ...prevCompositions };

      // Ensure value is correctly formatted (e.g., converting string to number for percentages)
      const formattedValue = field === "percentage" ? parseFloat(value) : value;

      // Directly update the specified field
      if (newCompositions[`comp${id}`]) {
        newCompositions[`comp${id}`][field] = formattedValue;
      }

      // For 'multiple', adjust the other composition's percentage if necessary
      if (type === "multiple" && field === "percentage") {
        const otherCompId = id === 1 ? 2 : 1; // Determine the other composition's id
        const totalPercentage =
          formattedValue + newCompositions[`comp${otherCompId}`].percentage;

        // If total exceeds 100%, adjust the other composition's percentage
        if (totalPercentage > 100) {
          newCompositions[`comp${otherCompId}`].percentage = Math.max(
            100 - formattedValue,
            0
          );
        }
      }

      return newCompositions;
    });
  };

  const handleInputChange = (e, key) => {
    setData({ ...data, [key]: e });
  };

  const handleSubmit = async () => {
    // TODO: validate form
    // if (
    //   (!adjacencyInput.file && !adjacencyInput.text) ||
    //   (!nodeFeatureInput.file && !nodeFeatureInput.text) ||
    //   !compositions.comp1.name ||
    //   !data[
    //     "Kappa BW-DCF(Bandwidth Dependent Die-electric Constant Fluctuation)"
    //   ] ||
    //   !data["Kappa-RSF"] ||
    //   !data["Membrane Thickness"] ||
    //   !data["Number of Lipid Per Layer"] ||
    //   !data["Number of Water Molecules"] ||
    //   !data["Pressure"] ||
    //   !data["Salt (moles per liter)"] ||
    //   !data["Temperature"]
    // ) {
    //   toast.error("Fill all the input fields");
    //   return;
    // }
    const formData = new FormData();

    inputs.forEach((input, index) => {
      // Determine the type (adjacency or nodeFeature) and lipid number (1 or 2)
      const type = index % 2 === 0 ? "adjacency" : "nodeFeature";
      const lipidNumber = index < 2 ? "1" : "2";

      // Append file and text for the current input
      if (input.file) {
        // Check if there's a file to append
        formData.append(`${type}File${lipidNumber}`, input.file);
      }
      formData.append(`${type}Text${lipidNumber}`, input.text);
    });

    // Add other data fields
    formData.append("type", type);
    formData.append("compositions", JSON.stringify(compositions));
    formData.append("data", JSON.stringify(data));

    try {
      setLoading(true);
      const response = await fetch("http://localhost:8000/test/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setPredictionValue(result.pred);
      // Handle result here
    } catch (error) {
      toast.error(
        `There was a problem with the fetch operation: ${error.message}`
      );
    }
    setLoading(false);
  };

  return (
    <div className="w-full h-full p-4">
      <Toaster />
      <div className="text-center mb-8">
        <RadioGroup
          name="radiogroup"
          size="large"
          defaultValue={type}
          onChange={(e) => handleTypeChange(e.target.value)}
          buttonStyle="solid"
        >
          <RadioButton value={"single"}>Single Composition</RadioButton>
          <RadioButton value={"multiple"}>Multiple Composition</RadioButton>
        </RadioGroup>
      </div>
      {type === "single" ? (
        <CompositionInput
          id={1}
          name={compositions.comp1.name}
          percentage={compositions.comp1.percentage}
          onCompositionChange={handleCompositionChange}
        />
      ) : (
        <>
          <CompositionInput
            id={1}
            showPercentage
            name={compositions.comp1.name}
            percentage={compositions.comp1.percentage}
            onCompositionChange={handleCompositionChange}
          />
          <CompositionInput
            id={2}
            showPercentage
            name={compositions.comp2.name}
            percentage={compositions.comp2.percentage}
            onCompositionChange={handleCompositionChange}
          />
        </>
      )}

      <div className="grid grid-cols-2 gap-4">
        {inputs.map((input, index) => {
          // Skip rendering for lipid-2 inputs if the type is "single"
          if (type === "single" && index > 1) return null;
          return (
            <BeadsBondsInput
              key={index} // Use index as key for simplicity; consider using unique ids for keys in real applications
              label={input.label}
              inputType={input.inputType}
              inputKey={index} // inputKey is now the index
              file={input.file}
              text={input.text}
              setInputType={(idx, type) => setInputsType(idx, type)}
              handleFileChange={(idx, file) => handleFileChange(idx, file)}
              handleTextChange={(idx, text) => handleTextChange(idx, text)}
            />
          );
        })}
      </div>

      <div className="grid grid-cols-2 gap-4 mt-6">
        {Object.keys(data).map((key) => (
          <div key={key}>
            <label
              htmlFor=""
              className="text-gray-800 tracking-tight font-semibold"
            >
              {key}
            </label>
            <InputNumber
              size="large"
              className="mt-1.5 w-full"
              value={data[key]}
              onChange={(e) => handleInputChange(e, key)}
            />
          </div>
        ))}
      </div>

      <div className="w-full mt-4 text-right">
        <button
          className="bg-blue-500 p-2 px-6 shadow rounded tracking-wider text-white font-medium"
          onClick={handleSubmit}
        >
          {loading ? (
            <CircularProgress size={"25px"} className="!text-white" />
          ) : (
            "Predict"
          )}
        </button>
      </div>
      {loading ? (
        <h1 className="mt-6 font-bold text-2xl">Fetching data...</h1>
      ) : (
        <>
          {predictionValue && (
            <div className="my-4 text-2xl gap-4 mt-8 flex font-mono items-center justify-center">
              <h1 className="text-gray-800">
                Prediction for{" "}
                <span className="text-gray-900 font-bold tracking-wide">
                  {compositions.comp1.name}
                </span>{" "}
                is:{" "}
              </h1>
              <p className="bg-violet-500 text-white font-bold p-2 px-4 rounded">
                {predictionValue.toFixed(3)}
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/**
 * Returns the initial state for the lipid compositions.
 * Sets up the initial names and percentages for compositions.
 *
 * @returns {Object} The initial composition state object.
 */

const initialCompositionState = () => ({
  comp1: { name: "", percentage: 100 },
  comp2: { name: "", percentage: 0 },
});

/**
 * Determines the state of compositions based on the selected type ('single' or 'multiple').
 * In 'single' mode, it sets comp1 percentage to 100%, and in 'multiple' mode, it sets both to 0%.
 *
 * @param {string} newType - The selected composition type.
 * @returns {Object} The updated composition state object.
 */

const compositionStateOnTypeChange = (newType) =>
  newType === "single"
    ? { comp1: { name: "", percentage: 100 } }
    : {
        comp1: { name: "", percentage: 0 },
        comp2: { name: "", percentage: 0 },
      };

/**
 * Calculates and returns the updated state for compositions.
 * In 'multiple' mode, it validates the total percentage does not exceed 100%.
 * Returns null if validation fails.
 *
 * @param {Object} compositions - The current compositions state.
 * @param {string} id - The composition identifier being updated.
 * @param {string} field - The field in the composition being updated ('name' or 'percentage').
 * @param {number|string} value - The new value for the field.
 * @param {string} type - The current selected type ('single' or 'multiple').
 * @returns {Object|null} The updated compositions state object or null if validation fails.
 */

const getUpdatedCompositions = (compositions, id, field, value, type) => {
  if (field === "percentage" && type === "multiple") {
    const totalPercentage =
      id === "comp1"
        ? value + compositions.comp2.percentage
        : compositions.comp1.percentage + value;
    if (totalPercentage > 100) {
      toast.error("Total percentage cannot exceed 100%");
      return null;
    }
  }
  return {
    ...compositions,
    [id]: { ...compositions[id], [field]: value },
  };
};

export default Prediction;
