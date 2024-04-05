import { CircularProgress } from "@mui/material";
import React, { useEffect, useState } from "react";

function ModelComparison() {
  const [images, setImages] = useState();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    (async function () {
      setLoading(true);
      const res = await fetch("http://localhost:8000/model_comp/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const data = await res.json();
      console.log(data);
      setImages(data);
      setLoading(false);
    })();
  }, []);

  return (
    <div className="p-4 w-full h-full">
      {loading ? (
        <div className="w-full h-full grid place-content-center">
          <CircularProgress></CircularProgress>
        </div>
      ) : (
        <div className="grid gap-4">
          {images &&
            Object.keys(images).map((val) => (
              <div key={val}>
                <img src={`data:image/png;base64,${images[val]}`} alt="" />
              </div>
            ))}
        </div>
      )}
    </div>
  );
}

export default ModelComparison;
