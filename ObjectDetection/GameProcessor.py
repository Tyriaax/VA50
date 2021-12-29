import cv2.cv2

from homography import *
from pawns_recognition import *
from cards_recognition import *
from GameBoard import *
from drawing import *

class GameProcessor:
    capEveryFrame = False
    homographymatrixfound = False

    def __init__(self, img, window_name):
        self.window_name = window_name

        self.height = img.shape[0]
        self.width = img.shape[1]

        self.gameBoard = GameBoard()
        self.pawnsRecognitionHelper = PawnsRecognitionHelper(self.height, self.width, self.gameBoard)
        self.cardsRecognitionHelper = CardsRecognitionHelper(self.height, self.width, self.gameBoard)

        cv2.setMouseCallback(self.window_name, self.ComputeMouseInput)

        self.list_board_coords = []
        self.actionPawnClicked = None

    def ComputeFrame(self, img):

        # If we are still waiting for the homography points
        if not self.homographymatrixfound:
            # When we have the 4 coordinates we can compute it
            if len(self.list_board_coords) == 4:
                # When we get the points for homography, set homographymatrix
                self.homographymatrix, self.coordinates = get_homography_matrix(img, np.array(self.list_board_coords), self.width, self.height)

                # Compute coordinates required for pawns and cards recognition helper
                self.cardsRecognitionHelper.GetScreenPortions(img, self.coordinates)
                self.pawnsRecognitionHelper.GetScreenPortion(img, self.coordinates)

                # Then directly compute the cards and their orientation
                self.cardsRecognitionHelper.ComputeFrame(img)
                
                self.homographymatrixfound = True
                img = cv2.warpPerspective(img, self.homographymatrix, (img.shape[1], img.shape[0]))
        else:
            # We need to apply homography here so that the frames are correctly computed
            img = cv2.warpPerspective(img, self.homographymatrix, (img.shape[1], img.shape[0]))

            # If we have a clicked Action Pawn to compute
            if self.actionPawnClicked is not None:

                # Different actions depending on the AP clicked
                if (self.actionPawnClicked.value <= 4):
                    self.pawnsRecognitionHelper.ComputeDetectivePawns(img)
                    print("Previous : \n", self.gameBoard.getPreviousDetectivePawns(), "\nCurrent: \n",
                          self.gameBoard.getDetectivePawns())
                elif (self.actionPawnClicked.value <= 7):
                    self.cardsRecognitionHelper.ComputeFrame(img)

                # We then check if the action pawns has been respected
                if (self.gameBoard.IsActionPawnRespected(self.actionPawnClicked.name)):
                    # TODO Why here Aurel ?
                    if (self.actionPawnClicked.value <= 4):
                        self.gameBoard.updatePreviousPawnsState()
                    else:
                        self.gameBoard.updatePreviousCardsState()

                    # We used the Action Pawns and now we remove it
                    print("Action Pawn Used")
                    self.pawnsRecognitionHelper.actionPawnUsed(self.actionPawnClicked)
                    self.gameBoard.nextTurn()

                    # If we used all the action pawns move to the next Game Event : Manhunt
                    if (len(self.gameBoard.getActionPawns()) == 0):
                        print("Turn Finished")
                        if (self.gameBoard.tryUpdateGameStatus(GameStates.GSAppealOfWitness)):
                            self.gameBoard.appealOfWitnesses(self.cardsRecognitionHelper.IsInLineOfSight(self.lastimg))
                            self.gameBoard.manhunt()
                else:
                    print("Action Pawn not Validated")

                # Either we were able to use the action pawn or not, it is not clicked anymore
                self.actionPawnClicked = None

        return img

    def DrawFrame(self, img):
        modifiedimg = img.copy()

        if not self.homographymatrixfound:
            modifiedimg = drawText(modifiedimg, "Selectionnez les quatre coins des 9 cartes",TextPositions.TPCenter)
            for coord in self.list_board_coords:
                cv2.circle(modifiedimg, coord, 10, (0, 255, 0), -1)
        elif self.gameBoard.getGameStatus() == GameStates.GSWaitingActionPawnsThrow:
            modifiedimg = self.pawnsRecognitionHelper.DrawZonesRectangles(modifiedimg, drawOffset=True)
            modifiedimg = self.cardsRecognitionHelper.DrawFrame(modifiedimg)
            modifiedimg = drawMultipleLinesOfText(modifiedimg, ["Appuyez sur P pour detecter les pions", "Ou sur C pour redetecter les cartes"], TextPositions.TPTopL)
        elif self.gameBoard.getGameStatus() == GameStates.GSUsingActionPawns:
            modifiedimg = self.cardsRecognitionHelper.DrawFrame(modifiedimg)
            modifiedimg = self.pawnsRecognitionHelper.DrawFrame(modifiedimg)
            modifiedimg = drawMultipleLinesOfText(modifiedimg, ["Realisez votre Action puis","Appuyez sur le jeton correspondant","Ou sur P pour redetecter les pions"], TextPositions.TPTopL)
            modifiedimg = drawPlayerAndTurn(modifiedimg, self.gameBoard.getCurrentPlayer(), self.gameBoard.getTurnCount())
        elif self.gameBoard.getGameStatus() == GameStates.GSAppealOfWitness:
            modifiedimg = self.cardsRecognitionHelper.DrawFrame(modifiedimg)
            modifiedimg = self.pawnsRecognitionHelper.DrawFrame(modifiedimg)
            modifiedimg = drawMultipleLinesOfText(modifiedimg, ["Retournez les cartes innocentees","Puis appuyez sur C"], TextPositions.TPTopL)
            modifiedimg = drawTurn(modifiedimg,self.gameBoard.getTurnCount())
        elif self.gameBoard.getGameStatus() == GameStates.GSGameOver:
            if (self.gameBoard.getDetectiveWins()):
                winnerstring = "Vous avez gagne"
            else:
                winnerstring = "Jack a gagne"
            modifiedimg = drawMultipleLinesOfText(modifiedimg, ["Partie terminee",winnerstring], TextPositions.TPCenter)


        return modifiedimg

    def ComputeInputs(self, img):
        continuebool = True

        key = cv2.waitKey(1) & 0xFF

        # Exit if Q or ESC pressed
        if key == ord('q') or key == 27:
            continuebool = False
        # If we press C for detect Cards
        if (key == ord('c') or self.capEveryFrame):
            # First this is here we check for victory, when we detect the cards after a ManHunt
            if (self.gameBoard.getDetectiveWins() or self.gameBoard.getJackWins()):
                self.gameBoard.tryUpdateGameStatus(GameStates.GSGameOver)

            elif (self.gameBoard.tryUpdateGameStatus(GameStates.GSWaitingActionPawnsThrow)):
                self.cardsRecognitionHelper.ComputeCards(img)
                self.cardsRecognitionHelper.ComputeFrame(img)

        # If we press P for detect Pawns
        if (key == ord('p') or self.capEveryFrame):

            if (self.gameBoard.tryUpdateGameStatus(GameStates.GSUsingActionPawns)):
                self.pawnsRecognitionHelper.ComputeFrame(img)
                self.gameBoard.printState()

        return continuebool

    def ComputeMouseInput(self,event,x,y,flags,params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # If we are still at the homography stage record the first 4 points coordinates
            if len(self.list_board_coords) < 4:
                self.list_board_coords.append([x, y])
            else:
                actionPawnIndex = self.pawnsRecognitionHelper.actionPawnClick([x,y])

                # If we click on a (remaining) action pawn bounding box
                if actionPawnIndex is not None:
                    actionPawnClicked = self.gameBoard.getActionPawns()[actionPawnIndex]
                    print("Action Pawn Clicked : " + actionPawnClicked)

                    self.actionPawnClicked = ActionPawns[actionPawnClicked]